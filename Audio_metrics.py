import numpy as np
import librosa
import scipy.signal as sps
from scipy.signal import hilbert

def compute_pitch(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=2048, sr=sr
    )
    return f0, voiced_flag, voiced_probs

def compute_jitter_shimmer(y, sr, f0, voiced_flag):
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    voiced_idx = np.where(voiced_flag)[0]
    if len(voiced_idx) < 3:
        return np.nan, np.nan
    T = 1/np.nan_to_num(f0[voiced_idx], nan=np.nan)
    T = T[~np.isnan(T)]
    if len(T) < 3:
        return np.nan, np.nan
    jitter = np.mean(np.abs(np.diff(T)))/np.mean(T)

    env = np.abs(hilbert(y))
    env_frames = librosa.util.frame(env, frame_length=1024, hop_length=512).max(axis=0)
    env_v = env_frames[:len(voiced_idx)]
    if len(env_v) < 3:
        return jitter, np.nan
    shimmer = np.mean(np.abs(np.diff(env_v)))/np.mean(env_v)
    return float(jitter), float(shimmer)

def compute_hnr(y, sr):
    y = y - np.mean(y)
    r = sps.correlate(y, y, mode='full')
    mid = len(r)//2
    r = r[mid:]
    r = r / np.max(r) if np.max(r) != 0 else r
    peaks, _ = sps.find_peaks(r, distance=10)
    if len(peaks) < 2:
        return np.nan
    peak = r[peaks[1]]
    hnr = 10*np.log10((peak)/(1-peak)) if 0 < peak < 1 else np.nan
    return float(hnr)

def compute_cpp(y, sr):
    win = np.hanning(len(y))
    yw = y*win
    spec = np.abs(np.fft.rfft(yw)) + 1e-9
    log_spectrum = np.log(spec)
    cepstrum = np.fft.irfft(log_spectrum)
    q = np.arange(len(cepstrum))/sr
    mask = (q>=0.002) & (q<=0.02)
    if not np.any(mask):
        return np.nan
    peak = np.max(cepstrum[mask])
    baseline = np.poly1d(np.polyfit(np.where(mask)[0], cepstrum[mask], deg=1))(np.where(mask)[0])
    prominence = float(peak - np.mean(baseline))
    return prominence

def summarize(y, sr):
    f0, vflag, vprob = compute_pitch(y, sr)
    pitch_mean = np.nanmean(f0)
    pitch_std = np.nanstd(f0)
    jitter, shimmer = compute_jitter_shimmer(y, sr, f0, vflag)
    hnr = compute_hnr(y, sr)
    cpp = compute_cpp(y, sr)
    return {
        "sr": sr,
        "duration_s": len(y)/sr,
        "pitch_mean_hz": float(pitch_mean) if pitch_mean==pitch_mean else np.nan,
        "pitch_std_hz": float(pitch_std) if pitch_std==pitch_std else np.nan,
        "jitter_rel": jitter,
        "shimmer_rel": shimmer,
        "HNR_dB": hnr,
        "CPP_simplified": cpp,
    }, f0, vflag, vprob
