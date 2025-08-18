import streamlit as st
import pandas as pd
import numpy as np
import librosa, soundfile as sf
import matplotlib.pyplot as plt
import io

from audio_metrics import summarize

# --- Page config ---
st.set_page_config(page_title="Phonalzye-Style Voice Analyzer – MVP")

st.title("Phonalzye-Style Voice Analyzer – MVP")
st.caption("Upload a voice clip (5–20s WAV/MP3) and get metrics + plots")

# --- File Upload ---
uploaded = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])

if uploaded is not None:
    # ✅ Proper loading from uploaded file
    data, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32")
    y = librosa.util.normalize(data)

    # --- Extract metrics ---
    report, f0, vflag, vprob = summarize(y, sr)

    # --- Show metrics ---
    st.subheader("Summary Metrics")
    st.json(report)

    # --- Waveform plot ---
    st.subheader("Waveform")
    fig1, ax1 = plt.subplots()
    t = np.arange(len(y)) / sr
    ax1.plot(t, y)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # --- Pitch contour ---
    st.subheader("Pitch Contour")
    fig2, ax2 = plt.subplots()
    times = librosa.frames_to_time(np.arange(len(f0)))
    ax2.plot(times, f0, linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("F0 (Hz)")
    if np.isfinite(np.nanmax(f0)):
        ax2.set_ylim(0, np.nanmax(f0) * 1.2)
    st.pyplot(fig2)="voice_metrics.csv", mime="text/csv")

else:
    st.info("Upload an audio file to begin.")
