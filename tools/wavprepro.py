from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
from scipy import signal
from scipy.io import wavfile
from hp import HP as hp
import tensorflow as tf
import pdb

''' 
modified from:
https://github.com/r9y9/deepvoice3_pytorch/blob/master/audio.py
https://github.com/andabi/voice-vector/blob/master/audio.py 
https://github.com/Kyubyong/dc_tts/blob/master/utils.py
'''
def load_spectrograms_npz(fpath,mode):
    fname = os.path.basename(fpath)
    #for some reasons windows started fucking with me :/
    if mode==1:
        mel_path=str(fpath).replace('.npz','_mel.npz').replace('\\','/').replace('//','/').replace("b'",'').replace("'",'')
        mag_path=str(fpath).replace('.npz','_mag.npz').replace('\\','/').replace('//','/').replace("b'",'').replace("'",'')
        mel=np.load(mel_path)
        mag=np.load(mag_path)
        return fname, mel['mel'].astype(np.float32), mag['mag'].astype(np.float32)
    elif mode==2:
        mel_path=str(fpath).replace('.npz','_mel.npz').replace('\\','/').replace('//','/').replace("b'",'').replace("'",'')
        mel=np.load(mel_path)
        return fname, mel['mel'].astype(np.float32)
    
def load_spectrograms(fpath):
    
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::hp.r, :]
    return fname, mel, mag

def get_spectrograms(fpath):

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1]) 

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          window='hann')

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (F, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (F, t)

    # amplitude to decibel
    mel = _amp_to_db(mel)
    mag = _amp_to_db(mag)

    # normalize
    mel = _normalize(mel)
    mag = _normalize(mag)


    # Transpose
    mel = mel.T.astype(np.float32)  # (T, F)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(mag):

    # transpose and ensuring mag
    mag = np.clip(mag.T,0,1)

    # de-normalize, convert back to linear magnitudes
    mag = _db_to_amp(_denormalize(mag)) 

    # apply sharpening factor
    mag = mag ** hp.sharpening_factor

    # wav reconstruction
    wav = griffin_lim(mag)

    # undo-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim out silent portions of signal
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):

    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t,n_fft=hp.n_fft,hop_length=hp.hop_length,window='hann')
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):

    return librosa.istft(spectrogram,hop_length=hp.hop_length, window="hann")


def _amp_to_db(x):
    min_level = np.exp(hp.min_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S):

    S = S - hp.min_db 
    S = S/np.max(S)        
    return np.clip(S, 1e-8, 1)

def _denormalize(S):
    S = S*(hp.ref_db-hp.min_db) 
    S = S + hp.min_db           
    return S

def save_wav(wav, path, sr):

    wavfile.write(path, sr, wav) 

