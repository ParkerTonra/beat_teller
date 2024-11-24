import os
import librosa
import sys
import numpy as np
import json

def analyze(file_path):
    try:
        # Commented out print statements to avoid extra output
        # print(f"Starting analysis on {file_path}")
        y, sr = librosa.load(file_path)
        # print(f"Loaded audio signal with shape: {y.shape} and sample rate: {sr}")

        # Calculate tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        print(json.dumps(tempo))

    except Exception as e:
        # Output the error as JSON
        error_result = {
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)
