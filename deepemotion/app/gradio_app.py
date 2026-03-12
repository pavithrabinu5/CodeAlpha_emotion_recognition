"""Gradio demo for DeepEmotion."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import gradio as gr

from src.predict import predict


def plot_waveform(audio: np.ndarray, sr: int) -> str:
    plt.figure(figsize=(8, 2))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix="_wave.png", delete=False)
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name


def plot_mel(audio: np.ndarray, sr: int) -> str:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix="_mel.png", delete=False)
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name


def predict_from_audio(audio_tuple, checkpoint):
    if audio_tuple is None:
        return "No audio provided", None, None
    sr, audio = audio_tuple
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        emotion, score = predict(tmp.name, checkpoint)
    wave_path = plot_waveform(audio, sr)
    mel_path = plot_mel(audio, sr)
    return f"{emotion} (confidence {score:.2f})", wave_path, mel_path


def launch(checkpoint: str, share: bool = False):
    iface = gr.Interface(
        fn=lambda audio: predict_from_audio(audio, checkpoint),
        inputs=[gr.Audio(type="numpy", label="Upload or record")],
        outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Waveform"), gr.Image(label="Mel Spectrogram")],
        title="DeepEmotion",
        description="Real-time speech emotion recognition",
        allow_flagging="never",
    )
    iface.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(args.checkpoint, share=args.share)
