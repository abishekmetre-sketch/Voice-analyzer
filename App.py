import streamlit as st
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import matplotlib.pyplot as plt

from audio_metrics import summarize

st.set_page_config(page_title="Phonalyze-Style Voice Analyzer (MVP)", layout="wide")

st.title("Phonalyze-Style Voice Analyzer — MVP")
st.caption("Upload a voice clip and get metrics + plots.")

uploaded = st.file_uploader("Upload WAV/MP3 (5–20s voice line)", type=["wav","mp3","m4a","ogg"])

if uploaded:
    y, sr = librosa.load(uploaded, sr=44100, mono=True)
    y = librosa.util.normalize(y)
    report, f0, vflag, vprob = summarize(y, sr)

    st.subheader("Summary Metrics")
    st.json(report)

    st.subheader("Waveform")
    fig1, ax1 = plt.subplots()
    t = np.arange(len(y))/sr
    ax1.plot(t, y)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    st.subheader("Pitch Contour")
    fig2, ax2 = plt.subplots()
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    ax2.plot(times, f0, linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("F0 (Hz)")
    if np.isfinite(np.nanmax(f0)):
        ax2.set_ylim(0, np.nanmax(f0)*1.2)
    st.pyplot(fig2)

    st.subheader("Spectrogram")
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    fig3, ax3 = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax3)
    fig3.colorbar(img, ax=ax3, format="%+2.f dB")
    st.pyplot(fig3)

    df = pd.DataFrame([report])
    st.download_button("Download metrics CSV", data=df.to_csv(index=False),
                       file_name="voice_metrics.csv", mime="text/csv")

else:
    st.info("Upload an audio file to begin.")
