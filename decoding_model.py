import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- regression loss  -------------------------
class RegressionLoss(nn.Module):
    def __init__(
        self,
        losses: List[str] = ["mse"],
        weights: Optional[List[float]] = None,
        huber_delta: float = 0.5,
    ):
        super().__init__()
        self.loss_fns = []
        self.weights = weights if weights is not None else [1.0] * len(losses)

        for loss in losses:
            if loss == "mse":
                self.loss_fns.append(nn.MSELoss(reduction="none"))
            elif loss == "l1":
                self.loss_fns.append(nn.L1Loss(reduction="none"))
            elif loss == "huber":
                self.loss_fns.append(nn.HuberLoss(reduction="none", delta=huber_delta))
            else:
                raise ValueError(f"Unsupported loss type: {loss}")

    def forward(
        self,
        predictions: torch.Tensor,                   # (B, T_pred, D)
        targets: torch.Tensor,                       # (B, T_tgt, D)
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = predictions.device
        total_loss = predictions.new_zeros(())

        # ---- align time dim ----
        T_pred = predictions.shape[1]
        T_tgt = targets.shape[1]
        T = min(T_pred, T_tgt)
        predictions = predictions[:, :T]
        targets = targets[:, :T]

        if targets.dtype != predictions.dtype:
            targets = targets.to(predictions.dtype)

        # ---- choose masking lengths ----
        lengths = None
        if target_lengths is not None and encoder_lengths is not None:
            lengths = torch.minimum(target_lengths.to(device), encoder_lengths.to(device))
        elif target_lengths is not None:
            lengths = target_lengths.to(device)
        elif encoder_lengths is not None:
            lengths = encoder_lengths.to(device)

        mask = None
        if lengths is not None:
            lengths = lengths.clamp(max=T).long()
            B, _, D = predictions.shape
            time_mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])
            mask = time_mask[:, :, None].expand(B, T, D).to(predictions.dtype)

        for fn, w in zip(self.loss_fns, self.weights):
            per_elem = fn(predictions, targets)  # (B,T,D)
            if mask is not None:
                denom = mask.sum().clamp(min=1)
                total_loss = total_loss + float(w) * (per_elem * mask).sum() / denom
            else:
                total_loss = total_loss + float(w) * per_elem.mean()

        return total_loss


# ------------------------- TransposeConv -> Conv -> GRU -> MLP -------------------------
def _conv1d_out_len(L: torch.Tensor, k: int, s: int, p: int, d: int = 1) -> torch.Tensor:
    # floor((L + 2p - d*(k-1) - 1)/s + 1)
    return torch.floor_divide(L + 2 * p - d * (k - 1) - 1, s) + 1


def _convtranspose1d_out_len(
    L: torch.Tensor, k: int, s: int, p: int, d: int = 1, out_p: int = 0
) -> torch.Tensor:
    # (L-1)*s - 2p + d*(k-1) + out_p + 1
    return (L - 1) * s - 2 * p + d * (k - 1) + out_p + 1


class ConvRNNArticulatoryRegressor(nn.Module):
    """
    Matches the figure:

      Neural Recordings
        -> TransposeConv1d (k=44, s=2)
        -> Conv1d          (k=1,  s=5)
        -> GRU             (d=512)
        -> MLP Decoder     (512 -> 256 -> 14)
        -> Articulatory Features

    Expected input:
      x:      (B, T, E)   where E = #electrodes/channels
      x_len:  (B,)        lengths in timesteps (before convs)

    Output:
      y:      (B, T', 14)
      y_len:  (B,)
      loss:   optional if targets are provided
    """

    def __init__(
        self,
        input_dim: int,                 # E
        out_dim: int = 14,
        # TransposeConv hyperparams (k=44, s=2)
        tconv_kernel: int = 44,
        tconv_stride: int = 2,
        tconv_padding: int = 0,
        tconv_out_padding: int = 0,
        tconv_channels: int = 256,      # choose an internal width (not shown in figure)
        # Conv hyperparams (k=1, s=5)
        conv_kernel: int = 1,
        conv_stride: int = 5,
        conv_padding: int = 0,
        conv_channels: int = 512,       # must be 512 to match "GRU (d=512)" in the figure
        # GRU
        gru_hidden: int = 512,
        gru_layers: int = 1,
        bidirectional: bool = False,    # figure implies 512->256->14, so default False
        gru_dropout: float = 0.0,
        # MLP decoder (512 -> 256 -> 14)
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.1,
        # regression loss
        reg_losses: List[str] = ["mse"],
        reg_loss_weights: Optional[List[float]] = None,
        huber_delta: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim

        # ---- TransposeConv: (B, E, T) -> (B, tconv_channels, T1)
        self.tconv = nn.ConvTranspose1d(
            in_channels=input_dim,
            out_channels=tconv_channels,
            kernel_size=tconv_kernel,
            stride=tconv_stride,
            padding=tconv_padding,
            output_padding=tconv_out_padding,
            bias=False,
        )
        self.tconv_bn = nn.BatchNorm1d(tconv_channels)
        self.tconv_act = nn.ReLU(inplace=True)

        # ---- Conv: (B, tconv_channels, T1) -> (B, 512, T2)
        self.conv = nn.Conv1d(
            in_channels=tconv_channels,
            out_channels=conv_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding=conv_padding,
            bias=False,
        )
        self.conv_bn = nn.BatchNorm1d(conv_channels)
        self.conv_act = nn.ReLU(inplace=True)

        # ---- GRU: (B, T2, 512) -> (B, T2, H*(1 or 2))
        self.bidirectional = bool(bidirectional)
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        rnn_out_dim = gru_hidden * (2 if self.bidirectional else 1)

        # ---- MLP decoder: rnn_out_dim -> 256 -> 14
        self.mlp = nn.Sequential(
            nn.LayerNorm(rnn_out_dim),
            nn.Linear(rnn_out_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, out_dim),
        )

        # ---- regression loss
        self.reg_loss = RegressionLoss(losses=reg_losses, weights=reg_loss_weights, huber_delta=huber_delta)

        # store conv geometry for length tracking
        self._tconv_k = int(tconv_kernel)
        self._tconv_s = int(tconv_stride)
        self._tconv_p = int(tconv_padding)
        self._tconv_op = int(tconv_out_padding)
        self._conv_k = int(conv_kernel)
        self._conv_s = int(conv_stride)
        self._conv_p = int(conv_padding)

    def _update_lengths(self, x_len: torch.Tensor) -> torch.Tensor:
        # after transpose conv
        x_len = _convtranspose1d_out_len(
            x_len, k=self._tconv_k, s=self._tconv_s, p=self._tconv_p, out_p=self._tconv_op
        )
        x_len = x_len.clamp_min(0)

        # after conv
        x_len = _conv1d_out_len(
            x_len, k=self._conv_k, s=self._conv_s, p=self._conv_p
        )
        x_len = x_len.clamp_min(0)
        return x_len.long()

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, E)
        x_len: torch.Tensor,                      # (B,)
        targets: Optional[torch.Tensor] = None,   # (B, T', 14) optional
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # (B, T, E) -> (B, E, T)
        x = x.transpose(1, 2)

        # TransposeConv
        x = self.tconv_act(self.tconv_bn(self.tconv(x)))  # (B, Ct, T1)

        # Conv (k=1, s=5)
        x = self.conv_act(self.conv_bn(self.conv(x)))     # (B, 512, T2)

        # lengths after convs
        y_len = self._update_lengths(x_len.to(x.device))

        # (B, 512, T2) -> (B, T2, 512)
        x = x.transpose(1, 2)

        # GRU
        y, _ = self.gru(x)  # (B, T2, H)

        # MLP decoder
        y = self.mlp(y)     # (B, T2, 14)

        if targets is None:
            return y, y_len, None

        # regression loss (masked if lengths provided)
        loss = self.reg_loss(
            predictions=y,
            targets=targets,
            encoder_lengths=y_len,
            target_lengths=target_lengths,
        )
        return y, y_len, loss


# ------------------------- quick sanity check -------------------------
if __name__ == "__main__":
    B, T, E = 2, 400, 64
    x = torch.randn(B, T, E)
    x_len = torch.tensor([400, 320])

    model = ConvRNNArticulatoryRegressor(input_dim=E)
    y, y_len, _ = model(x, x_len)
    print("y:", y.shape, "y_len:", y_len)
