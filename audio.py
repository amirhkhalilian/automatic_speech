import os
from os import path
import argparse

import numpy as np
import pandas as pd
import json
from pprint import pprint

from scipy.signal import resample_poly

from pyhypnos.loadingtools import get_zoom_from_mat
from pyhypnos.loadingtools import get_events

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the audio data
        audio_data = wav_file.readframes(wav_file.getnframes())
        # Convert the binary data to a NumPy array with int16 dtype
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        n_channels = wav_file.getnchannels()
        if n_channels > 1:
            audio_array = np.reshape(audio_array, (-1, n_channels))
    return audio_array, wav_file.getframerate()

def write_wav_file(file_path, audio_array, sample_rate):
    with wave.open(file_path, 'wb') as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(2)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())

def get_audio_name(subj):
    dr = os.path.join('EcogData',
                      subj, f'{subj}_SpeechOutput')
    if subj in ['NY758', 'NY765']:
        fn = os.path.join(dr, f'{subj}_SpeechOutput.WAV')
    elif subj in ['NY798', 'NY829', 'NY857', 'NY869']:
        fn = os.path.join(dr, f'{subj}_SpeechOutput_uni.wav')
    return dr, fn

def _check_events_zoom(subj, task,
                       zoom_fs=44100,
                       events_fs=512,
                       flag_response_lock=False,
                       epoch_start = 0,
                       epoch_end = 44100):
    subj_dir = os.path.join(f'EcogData/{subj}')
    evnt_dir = os.path.join(f'EcogData/',
                            subj,
                            'analysis',
                            task,
                            'Events.mat')
    zoom_dir = os.path.join(f'EcogData/',
                            subj,
                            'data',
                            task,
                            'zoom.mat')

    zoom = get_zoom_from_mat(zoom_dir)
    zoom = zoom.squeeze()
    print(f"loaded zoom file of shpae: {zoom.shape}")
    print(f"zoom length: {zoom.shape[0]/zoom_fs} sec")
    stamps = get_events(evnt_dir)
    if not flag_response_lock:
        onset_512 = stamps['onset']
    else:
        onset_512 = stamps['onset_r']
    onset = (onset_512/events_fs*zoom_fs).astype(int)
    Ntrials = stamps['trials2use'].shape[0]
    Nsamples = np.abs(epoch_start) + epoch_end
    data = np.empty((Ntrials, Nsamples), dtype=float)
    ind_st = onset + epoch_start
    ind_en = onset + epoch_end
    for i, tr in enumerate(stamps['trials2use']):
        if ind_st[tr]<0:
            data_tmp = zoom[0:ind_en[tr]]
            pad = np.zeros((Nsamples-data_tmp.shape[0]))
            data[i,:] = np.concatenate((pad, data_tmp), axis=0)
        elif ind_en[tr]>zoom.shape[0]:
            data_tmp = zoom[ind_st[tr]:]
            pad = np.zeros((Nsamples-data_tmp.shape[0]))
            data[i,:] = np.concatenate((data_tmp, pad), axis=0)
        else:
            data[i,:] = zoom[np.arange(ind_st[tr],ind_en[tr])]
    return data

def get_ema_features(subj, task,
                     zoom_fs=44100,
                     events_fs=512,
                     flag_response_lock=False,
                     epoch_start = 0,
                     epoch_end = 44100):
    '''
    function to load the audio from zoom
    convert to sparc features and save
    '''
    from sparc import load_model
    print(f"working on subj:{subj} task:{task}")
    zoom_audio = _check_events_zoom(subj, task,
                                    zoom_fs=zoom_fs,
                                    events_fs=events_fs,
                                    flag_response_lock=flag_response_lock,
                                    epoch_start = epoch_start,
                                    epoch_end = epoch_end)
    print(f"loaded zoom of shape: {zoom_audio.shape}")
    print(f"n_trials: {zoom_audio.shape[0]}")
    print(f"time: {zoom_audio.shape[1]/zoom_fs}")

    coder = load_model("en", device= "cpu", use_penn=True)
    ema_all = []
    for i in range(zoom_audio.shape[0]):
        zoom_in = resample_poly(zoom_audio[i,:],16000,zoom_fs)
        code = coder.encode(zoom_in)
        m = min(code['ema'].shape[0], code['pitch'].shape[0], code['loudness'].shape[0])
        ema_i = np.concatenate([code['ema'][:m,:],
                                code['pitch'][:m,:],
                                code['loudness'][:m,:]], axis=1)
        ema_all.append(ema_i[None,...])
    ema_all = np.concatenate(ema_all, axis=0)
    print(f"processed ema shape:{ema_all.shape}")
    ema_dir = os.path.join(f'EcogData/',
                            subj,
                            'data',
                            task,
                            'ema.npz')
    np.savez(ema_dir, ema=ema_all,
             epoch_start=epoch_start,
             epoch_end=epoch_end,
             zoom_fs=zoom_fs)

def _preprocess_ema():
    subjs = ['NY749', 'NY758', 'NY765', 'NY798', 'NY829', 'NY857', 'NY869']
    tasks = ['SpeechOutput_C', 'SpeechOutput_D', 'SpeechOutput_M']
    for subj in subjs:
        for task in tasks:
            get_ema_features(subj, task,
                             zoom_fs=44100,
                             events_fs=512,
                             flag_response_lock=False,
                             epoch_start = int(-0.28*44100),
                             epoch_end = 44100)

if __name__=="__main__":
    _preprocess_ema()


