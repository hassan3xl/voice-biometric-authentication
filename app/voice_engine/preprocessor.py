import base64
import io
import math
import wave
import numpy as np
import soundfile as sf
from scipy import signal


class AudioPreprocessor:
    """Preprocesses acoustic signals for voice biometric authentication.
    
    Handles audio decoding, 16kHz resampling, mono conversion,
    pre-emphasis filtering, Voice Activity Detection (VAD),
    and Signal-to-Noise Ratio (SNR) estimation.
    """

    TARGET_SAMPLE_RATE = 16000
    PRE_EMPHASIS_COEFF = 0.97
    FRAME_LENGTH_MS = 25  # 25 ms window
    FRAME_STEP_MS = 10    # 10 ms hop size
    MIN_SNR_DB = 8.0      # Minimum acceptable SNR for banking verification
    MIN_SPEECH_DURATION_SEC = 0.6  # Minimum voiced duration required

    @classmethod
    def load_audio_from_bytes(cls, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Loads audio from raw bytes (WAV, RIFF, or container) into float32 array [-1.0, 1.0] and sample rate."""
        # Check if base64 encoded
        if audio_bytes.startswith(b'data:audio') or b';base64,' in audio_bytes[:50]:
            try:
                base64_data = audio_bytes.split(b',', 1)[1]
                audio_bytes = base64.b64decode(base64_data)
            except Exception:
                pass
        elif not audio_bytes.startswith(b'RIFF') and not audio_bytes.startswith(b'OggS'):
            try:
                # Attempt base64 decode if plain base64 string
                audio_bytes = base64.b64decode(audio_bytes)
            except Exception:
                pass

        try:
            # Use soundfile first
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        except Exception:
            # Fallback to standard library wave
            try:
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    raw_frames = wf.readframes(n_frames)
                    
                    if sampwidth == 2:  # 16-bit PCM
                        data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:  # 8-bit unsigned
                        data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    elif sampwidth == 4:  # 32-bit float or int
                        data = np.frombuffer(raw_frames, dtype=np.float32)
                    else:
                        data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                    if n_channels > 1:
                        data = data.reshape(-1, n_channels)
            except Exception as e:
                raise ValueError(f"Unable to parse audio stream: {e}")

        # Convert to mono if multi-channel
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # Resample to TARGET_SAMPLE_RATE if different
        if sr != cls.TARGET_SAMPLE_RATE:
            num_target_samples = int(len(data) * cls.TARGET_SAMPLE_RATE / sr)
            if num_target_samples > 0:
                data = signal.resample(data, num_target_samples)
                sr = cls.TARGET_SAMPLE_RATE

        return data.astype(np.float32), sr

    @classmethod
    def apply_pre_emphasis(cls, signal_data: np.ndarray, alpha: float = PRE_EMPHASIS_COEFF) -> np.ndarray:
        """High-pass filter: y[t] = x[t] - alpha * x[t-1] to amplify high-frequency formants."""
        if len(signal_data) < 2:
            return signal_data
        return np.append(signal_data[0], signal_data[1:] - alpha * signal_data[:-1])

    @classmethod
    def compute_snr_and_quality(cls, signal_data: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict:
        """Estimates Signal-to-Noise Ratio (SNR), clipping percentage, and speech presence."""
        if len(signal_data) == 0:
            return {'snr_db': 0.0, 'clipping_pct': 0.0, 'is_acceptable': False, 'reason': 'Empty audio'}

        # Calculate clipping percentage (samples near extreme +/- 0.99)
        clipping_samples = np.sum(np.abs(signal_data) >= 0.99)
        clipping_pct = float(clipping_samples / len(signal_data)) * 100.0

        # Calculate overall RMS energy
        rms = float(np.sqrt(np.mean(signal_data ** 2)))
        if rms < 0.003:
            return {
                'snr_db': 0.0,
                'clipping_pct': clipping_pct,
                'is_acceptable': False,
                'reason': 'No speech detected (silent audio)'
            }

        # Frame-level energy estimation
        frame_len = int(sr * 0.025)
        hop_len = int(sr * 0.010)

        num_frames = max(1, (len(signal_data) - frame_len) // hop_len + 1)
        frame_energies = []
        for i in range(num_frames):
            frame = signal_data[i * hop_len : i * hop_len + frame_len]
            frame_energies.append(np.sum(frame ** 2) / max(1, len(frame)))

        frame_energies = np.array(frame_energies)
        if len(frame_energies) == 0 or np.max(frame_energies) == 0:
            return {'snr_db': 0.0, 'clipping_pct': clipping_pct, 'is_acceptable': False, 'reason': 'Silent audio'}

        # Sort frame energies to estimate background noise floor (bottom 20%) and speech signal (top 30%)
        sorted_energies = np.sort(frame_energies)
        noise_floor_energy = np.mean(sorted_energies[:max(1, int(len(sorted_energies) * 0.20))]) + 1e-12
        signal_peak_energy = np.mean(sorted_energies[-max(1, int(len(sorted_energies) * 0.30)):]) + 1e-12

        snr_db = float(10.0 * np.log10(signal_peak_energy / noise_floor_energy))

        # Check for adequate dynamic range or solid active speech energy
        is_acceptable = (clipping_pct < 5.0) and ((snr_db >= 3.0) or (rms >= 0.02)) and (rms >= 0.003)
        reason = "Optimal audio quality" if is_acceptable else (
            "Microphone audio clipping" if clipping_pct >= 5.0 else
            ("No speech detected (too quiet)" if rms < 0.003 else "Background noise too high")
        )

        return {
            'snr_db': round(snr_db, 2),
            'clipping_pct': round(clipping_pct, 2),
            'is_acceptable': is_acceptable,
            'reason': reason
        }

    @classmethod
    def apply_vad(cls, signal_data: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, float]:
        """Performs Voice Activity Detection (VAD) using adaptive energy thresholding.
        Returns: (voiced_signal, speech_duration_seconds)
        """
        if len(signal_data) == 0:
            return signal_data, 0.0

        frame_len = int(sr * cls.FRAME_LENGTH_MS / 1000)
        hop_len = int(sr * cls.FRAME_STEP_MS / 1000)

        num_frames = max(1, (len(signal_data) - frame_len) // hop_len + 1)
        frames = []
        energies = []
        zcrs = []

        for i in range(num_frames):
            idx = i * hop_len
            frame = signal_data[idx : idx + frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            frames.append(frame)

            # Short-time Energy
            energy = np.sum(frame ** 2) / len(frame)
            energies.append(energy)

            # Zero Crossing Rate
            zcr = 0.5 * np.mean(np.abs(np.diff(np.sign(frame))))
            zcrs.append(zcr)

        energies = np.array(energies)
        zcrs = np.array(zcrs)

        # Dynamic energy threshold (adaptive percentile based on noise floor)
        noise_floor = np.percentile(energies, 20)
        peak_energy = np.percentile(energies, 95)
        
        # Energy threshold between noise floor and peak
        energy_threshold = max(0.0005, noise_floor + 0.15 * (peak_energy - noise_floor))

        voiced_indices = np.where(energies > energy_threshold)[0]

        if len(voiced_indices) == 0:
            return signal_data, 0.0

        # Smooth voiced segments with morphological dilation (hangover scheme)
        start_frame = max(0, voiced_indices[0] - 2)
        end_frame = min(num_frames, voiced_indices[-1] + 3)

        start_sample = start_frame * hop_len
        end_sample = min(len(signal_data), end_frame * hop_len + frame_len)

        voiced_signal = signal_data[start_sample:end_sample]
        speech_duration = len(voiced_signal) / sr

        return voiced_signal, speech_duration

    @classmethod
    def frame_signal(cls, signal_data: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        """Splits signal into overlapping Hamming-windowed frames of shape (num_frames, frame_length)."""
        frame_len = int(sr * cls.FRAME_LENGTH_MS / 1000)  # 400 for 16kHz
        hop_len = int(sr * cls.FRAME_STEP_MS / 1000)      # 160 for 16kHz

        signal_len = len(signal_data)
        if signal_len < frame_len:
            signal_data = np.pad(signal_data, (0, frame_len - signal_len))
            signal_len = frame_len

        num_frames = 1 + int(math.ceil((signal_len - frame_len) / hop_len))
        pad_signal_len = (num_frames - 1) * hop_len + frame_len
        pad_signal = np.pad(signal_data, (0, pad_signal_len - signal_len))

        # Vectorized frame extraction
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * hop_len, hop_len), (frame_len, 1)).T
        
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # Apply Hamming window
        hamming_win = np.hamming(frame_len)
        return frames * hamming_win

    # Method aliases for developer convenience
    load_and_resample = load_audio_from_bytes
    voice_activity_detection = apply_vad
    compute_signal_to_noise_ratio = compute_snr_and_quality
