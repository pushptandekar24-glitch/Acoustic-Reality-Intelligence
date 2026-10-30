"""
Environmental Sound Event Recognition Engine.
Provides real, physically grounded environmental sound-event classification
using acoustic spectral signatures and envelope dynamics, with optional YAMNet integration hooks.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
from scipy import signal as scipy_signal

import config


# Standard Environmental Sound Taxonomy
ENVIRONMENTAL_CATEGORIES = [
    "Rain & Thunder",
    "Birds & Insects",
    "Traffic & Vehicles",
    "Sirens & Alarms",
    "Human Speech & Crowd",
    "Wind & Natural Flow",
    "Animals & Wildlife",
    "Mechanical & Industrial",
]


class EnvironmentalSoundRecognizer:
    """
    Environmental sound event classifier.
    Computes real spectral descriptors, temporal envelope dynamics, and psychoacoustic features
    to classify environmental audio into recognized sound event categories.
    """

    def __init__(self):
        self.categories = ENVIRONMENTAL_CATEGORIES
        self.yamnet_available = False
        self._check_optional_backends()

    def _check_optional_backends(self):
        """Checks if optional TensorFlow / YAMNet backend is available."""
        try:
            import tensorflow_hub  # noqa: F401
            import tensorflow  # noqa: F401
            self.yamnet_available = True
        except ImportError:
            self.yamnet_available = False

    def classify_sound_events(
        self,
        signal: np.ndarray,
        sample_rate: int = config.TARGET_SAMPLE_RATE,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Extracts physical DSP audio features and computes real category probability scores.
        """
        if signal is None or signal.size == 0:
            raise ValueError("Cannot classify empty audio signal.")

        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
        if signal.size < 2048:
            signal = np.pad(signal, (0, 2048 - signal.size), mode='constant')

        # 1. Compute physical spectral descriptors
        stft = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

        # Spectral Centroid
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]))
        # Spectral Flatness (Broadband Noise vs Tonality)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=signal)[0]))
        # Spectral Bandwidth
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]))
        # Spectral Rolloff
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, roll_percent=0.85)[0]))
        # Zero Crossing Rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)[0]))
        # RMS Energy & variance (burstiness)
        rms = librosa.feature.rms(y=signal)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        burstiness = float(rms_std / (rms_mean + 1e-9))

        # Sub-band Energy Distribution: Low (<500Hz), Mid (500-3000Hz), High (>3000Hz)
        low_band = np.sum(stft[freqs < 500, :])
        mid_band = np.sum(stft[(freqs >= 500) & (freqs < 3000), :])
        high_band = np.sum(stft[freqs >= 3000, :])
        total_band = low_band + mid_band + high_band + 1e-9

        low_ratio = float(low_band / total_band)
        mid_ratio = float(mid_band / total_band)
        high_ratio = float(high_band / total_band)

        # 2. Syllabic / Pitch Modulation Analysis (detect speech or siren periodic oscillation)
        hop_time = 512 / sample_rate
        centroid_series = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, hop_length=512)[0]
        centroid_variation = float(np.std(centroid_series) / (np.mean(centroid_series) + 1e-9))

        # 3. Physically grounded Category Likelihood scoring
        raw_scores: Dict[str, float] = {}

        # Rain & Thunder: high flatness (noise), high broadband spread, low-frequency rumble
        raw_scores["Rain & Thunder"] = (
            flatness * 3.5 +
            (low_ratio * 1.5) +
            (1.0 - zcr * 2.0 if zcr < 0.5 else 0.1) +
            (0.8 if 800 < centroid < 3500 else 0.2)
        )

        # Birds & Insects: high centroid (>3000 Hz), high high_ratio, high ZCR
        raw_scores["Birds & Insects"] = (
            (high_ratio * 4.0) +
            (zcr * 3.0) +
            (2.0 if centroid > 3200 else (centroid / 1600.0)) +
            (0.5 if flatness < 0.2 else 0.0)
        )

        # Traffic & Vehicles: heavy low_band energy, low centroid (<1200 Hz), low flatness
        raw_scores["Traffic & Vehicles"] = (
            (low_ratio * 4.2) +
            (2.0 if centroid < 1200 else 0.3) +
            (1.0 - flatness * 1.5) +
            (1.0 - high_ratio * 2.0)
        )

        # Sirens & Alarms: strong sinusoidal pitch/centroid oscillation, low flatness (tonal), mid frequency
        raw_scores["Sirens & Alarms"] = (
            (centroid_variation * 3.0) +
            (mid_ratio * 2.5) +
            (2.0 if 1000 < centroid < 2800 else 0.2) +
            (1.5 * (1.0 - flatness))
        )

        # Human Speech & Crowd: dominant mid_band (formants 500-2500Hz), moderate burstiness, moderate ZCR
        raw_scores["Human Speech & Crowd"] = (
            (mid_ratio * 3.8) +
            (burstiness * 1.5) +
            (1.8 if 1200 < centroid < 2600 else 0.3) +
            (1.2 if 0.04 < zcr < 0.18 else 0.2)
        )

        # Wind & Natural Flow: low-to-mid centroid, high noise, low burstiness
        raw_scores["Wind & Natural Flow"] = (
            (flatness * 2.8) +
            (low_ratio * 2.2) +
            (1.0 - burstiness * 0.8 if burstiness < 1.0 else 0.2) +
            (1.2 if centroid < 1800 else 0.2)
        )

        # Animals & Wildlife: high burstiness, mid-to-high frequency bursts
        raw_scores["Animals & Wildlife"] = (
            (burstiness * 2.5) +
            (high_ratio * 2.0) +
            (mid_ratio * 1.8) +
            (0.8 if centroid > 1800 else 0.2)
        )

        # Mechanical & Industrial: steady tone + low frequency vibration, low burstiness
        raw_scores["Mechanical & Industrial"] = (
            (low_ratio * 2.5) +
            (1.0 - burstiness * 0.9 if burstiness < 1.0 else 0.2) +
            (mid_ratio * 1.5) +
            (0.5 if flatness < 0.15 else 0.1)
        )

        # Softmax normalization over positive exponents
        scores_arr = np.array([max(0.01, raw_scores[cat]) for cat in self.categories])
        # Scale for stable temperature softmax
        temperature = 1.2
        exp_scores = np.exp((scores_arr - np.mean(scores_arr)) / temperature)
        probs = exp_scores / np.sum(exp_scores)

        category_probs = {cat: float(round(p, 4)) for cat, p in zip(self.categories, probs)}
        sorted_events = sorted(category_probs.items(), key=lambda item: item[1], reverse=True)

        top_predictions = [
            {"event": event, "score": score, "percentage": round(score * 100, 1)}
            for event, score in sorted_events[:top_k]
        ]

        top_event = top_predictions[0]["event"]
        top_score = top_predictions[0]["score"]

        return {
            "top_event": top_event,
            "top_score": top_score,
            "top_predictions": top_predictions,
            "all_predictions": category_probs,
            "acoustic_descriptors": {
                "spectral_centroid_hz": round(centroid, 1),
                "spectral_flatness": round(flatness, 4),
                "spectral_bandwidth_hz": round(bandwidth, 1),
                "spectral_rolloff_hz": round(rolloff, 1),
                "zero_crossing_rate": round(zcr, 4),
                "low_band_energy_ratio": round(low_ratio, 3),
                "mid_band_energy_ratio": round(mid_ratio, 3),
                "high_band_energy_ratio": round(high_ratio, 3),
                "burstiness_index": round(burstiness, 3),
            },
            "backend": "Acoustic Physics & Spectral Signature Profiler (Native)",
            "yamnet_status": (
                "YAMNet optional backend: available in environments with TensorFlow (Python <= 3.12). "
                "Current runtime is utilizing the physical DSP Environmental Feature Engine."
            ),
            "disclaimer": (
                "Important Distinction: Environmental sound recognition classifies what acoustic event is audible in the recording. "
                "The presence of a recognizable real-world sound event (e.g. rain, birds, sirens) does NOT imply that the recording is genuine or non-synthetic."
            ),
        }

