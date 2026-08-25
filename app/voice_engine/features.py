import math
import numpy as np
from scipy.fftpack import dct


class AcousticFeatureExtractor:
    """Extracts Mel-Frequency Cepstral Coefficients (MFCCs),
    spectral moments, and fundamental pitch dynamics for voice biometrics.
    """

    DEFAULT_N_MFCC = 20
    DEFAULT_N_FFT = 512
    DEFAULT_N_MELS = 40
    SAMPLE_RATE = 16000

    @classmethod
    def hz_to_mel(cls, hz: float | np.ndarray) -> float | np.ndarray:
        """Converts frequency in Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @classmethod
    def mel_to_hz(cls, mel: float | np.ndarray) -> float | np.ndarray:
        """Converts Mel scale to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def get_filterbanks(cls, n_mels: int = DEFAULT_N_MELS, n_fft: int = DEFAULT_N_FFT, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
        low_freq_mel = cls.hz_to_mel(0)
        high_freq_mel = cls.hz_to_mel(sr / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = cls.mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
        for m in range(1, n_mels + 1):
            f_m_minus = int(bin_points[m - 1])
            f_m = int(bin_points[m])
            f_m_plus = int(bin_points[m + 1])

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        return fbank

    @classmethod
    def compute_deltas(cls, feat_matrix: np.ndarray, N: int = 2) -> np.ndarray:
        r"""Computes dynamic regression delta coefficients over time frames.
        Formula: d_t = \sum_{n=1}^N n * (c_{t+n} - c_{t-n}) / (2 * \sum_{n=1}^N n^2)
        """
        num_frames = feat_matrix.shape[0]
        denom = 2 * sum(i ** 2 for i in range(1, N + 1))
        padded = np.pad(feat_matrix, ((N, N), (0, 0)), mode='edge')
        deltas = np.zeros_like(feat_matrix)

        for t in range(num_frames):
            acc = np.zeros(feat_matrix.shape[1])
            for n in range(1, N + 1):
                acc += n * (padded[t + N + n] - padded[t + N - n])
            deltas[t] = acc / denom

        return deltas

    @classmethod
    def extract_mfcc(
        cls,
        frames: np.ndarray,
        n_mfcc: int = DEFAULT_N_MFCC,
        n_fft: int = DEFAULT_N_FFT,
        n_mels: int = DEFAULT_N_MELS,
        sr: int = SAMPLE_RATE,
        apply_cmvn: bool = True
    ) -> np.ndarray:
        """Extracts 60-dimensional MFCC vectors (Static + Delta + Delta-Delta) for windowed frames."""
        if frames.ndim == 1:
            from app.voice_engine.preprocessor import AudioPreprocessor
            frames = AudioPreprocessor.frame_signal(frames, sr=sr)

        # 1. Compute Short-Time Fourier Transform Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, n_fft))  # Shape: (num_frames, n_fft // 2 + 1)
        pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

        # 2. Apply Mel Filterbanks
        fbanks = cls.get_filterbanks(n_mels=n_mels, n_fft=n_fft, sr=sr)
        filter_banks = np.dot(pow_frames, fbanks.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        # 3. Discrete Cosine Transform (DCT-II)
        raw_mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

        # 4. Cepstral Mean & Variance Normalization (CMVN)
        if apply_cmvn and raw_mfcc.shape[0] > 1:
            mean = np.mean(raw_mfcc, axis=0)
            std = np.std(raw_mfcc, axis=0) + 1e-8
            raw_mfcc = (raw_mfcc - mean) / std

        # 5. Dynamic temporal deltas
        deltas = cls.compute_deltas(raw_mfcc, N=2)
        delta_deltas = cls.compute_deltas(deltas, N=2)

        # Full 60-D acoustic feature matrix (num_frames, 3 * n_mfcc)
        full_mfcc = np.hstack([raw_mfcc, deltas, delta_deltas])
        return full_mfcc

    @classmethod
    def estimate_pitch_f0(cls, signal_data: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """Estimates Fundamental Frequency (F0), pitch contours, jitter, and shimmer via autocorrelation."""
        if len(signal_data) < int(sr * 0.05):
            return {'mean_f0_hz': 0.0, 'f0_std': 0.0, 'jitter_pct': 0.0, 'shimmer_pct': 0.0, 'voiced_ratio': 0.0}

        frame_len = int(sr * 0.040)  # 40 ms window for reliable low pitch tracking
        hop_len = int(sr * 0.015)    # 15 ms hop
        min_f0 = 65.0   # Hz (Low Male)
        max_f0 = 400.0  # Hz (High Female / Child)

        min_lag = int(sr / max_f0)
        max_lag = int(sr / min_f0)

        num_frames = max(1, (len(signal_data) - frame_len) // hop_len + 1)
        f0_list = []
        amp_list = []

        for i in range(num_frames):
            frame = signal_data[i * hop_len : i * hop_len + frame_len]
            if len(frame) < frame_len:
                continue

            # Normalized Autocorrelation
            frame_centered = frame - np.mean(frame)
            norm_factor = np.sum(frame_centered ** 2)
            if norm_factor < 1e-6:
                continue

            autocorr = np.correlate(frame_centered, frame_centered, mode='full')
            autocorr = autocorr[len(frame_centered) - 1 :] / norm_factor

            # Find peak in [min_lag, max_lag]
            lag_search_window = autocorr[min_lag:max_lag]
            if len(lag_search_window) == 0:
                continue

            peak_idx = np.argmax(lag_search_window) + min_lag
            peak_val = autocorr[peak_idx]

            # Voiced threshold
            if peak_val > 0.35:
                f0 = sr / peak_idx
                f0_list.append(f0)
                amp_list.append(np.max(frame) - np.min(frame))

        if len(f0_list) < 3:
            return {'mean_f0_hz': 120.0, 'f0_std': 0.0, 'jitter_pct': 0.0, 'shimmer_pct': 0.0, 'voiced_ratio': 0.0}

        f0_arr = np.array(f0_list)
        amp_arr = np.array(amp_list)

        # Jitter (Period perturbation percentage within stable voiced frames)
        periods = 1.0 / f0_arr
        valid_period_diffs = []
        for i in range(1, len(f0_arr)):
            if abs(f0_arr[i] - f0_arr[i - 1]) / f0_arr[i - 1] < 0.35:
                valid_period_diffs.append(abs(periods[i] - periods[i - 1]))

        if valid_period_diffs:
            jitter_pct = float(np.mean(valid_period_diffs) / (np.mean(periods) + 1e-9)) * 100.0
        else:
            jitter_pct = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9)) * 100.0

        # Shimmer (Amplitude perturbation percentage within voiced frames)
        valid_amp_diffs = []
        for i in range(1, len(amp_arr)):
            if amp_arr[i] > 1e-4 and amp_arr[i - 1] > 1e-4:
                valid_amp_diffs.append(abs(amp_arr[i] - amp_arr[i - 1]))

        if valid_amp_diffs:
            shimmer_pct = float(np.mean(valid_amp_diffs) / (np.mean(amp_arr) + 1e-9)) * 100.0
        else:
            shimmer_pct = float(np.mean(np.abs(np.diff(amp_arr))) / (np.mean(amp_arr) + 1e-9)) * 100.0

        mean_f0 = float(np.mean(f0_arr))
        f0_std = float(np.std(f0_arr))
        voiced_ratio = float(len(f0_list) / max(1, num_frames))

        return {
            'mean_f0_hz': round(mean_f0, 2),
            'f0_std': round(f0_std, 2),
            'jitter_pct': round(jitter_pct, 3),
            'shimmer_pct': round(shimmer_pct, 3),
            'voiced_ratio': round(voiced_ratio, 2)
        }

    @classmethod
    def compute_spectral_moments(cls, frames: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = DEFAULT_N_FFT) -> dict:
        """Computes Spectral Centroid, Rolloff, and Spectral Flatness."""
        if frames.ndim == 1:
            from app.voice_engine.preprocessor import AudioPreprocessor
            frames = AudioPreprocessor.frame_signal(frames, sr=sr)

        # Filter low-energy frames so background pauses don't skew spectral moments
        frame_energies = np.sum(frames ** 2, axis=1)
        max_energy = np.max(frame_energies) if len(frame_energies) > 0 else 0
        if max_energy > 1e-6:
            active_mask = frame_energies >= (0.01 * max_energy)
            if np.any(active_mask):
                frames = frames[active_mask]

        mag_spec = np.absolute(np.fft.rfft(frames, n_fft))  # (num_frames, freq_bins)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Spectral Centroid: \sum f * S(f) / \sum S(f)
        sum_spec = np.sum(mag_spec, axis=1) + 1e-9
        centroids = np.sum(mag_spec * freqs, axis=1) / sum_spec
        mean_centroid = float(np.mean(centroids))

        # Spectral Flatness: Geometric Mean / Arithmetic Mean (Wiener entropy)
        # Synthetic TTS speech often has abnormal spectral flatness
        log_spec = np.log(mag_spec + 1e-12)
        geom_mean = np.exp(np.mean(log_spec, axis=1))
        arith_mean = np.mean(mag_spec, axis=1) + 1e-12
        flatness = geom_mean / arith_mean
        mean_flatness = float(np.mean(flatness))

        # Spectral Rolloff: Frequency below which 85% of power spectral energy resides
        cum_energy = np.cumsum(mag_spec ** 2, axis=1)
        total_energy = cum_energy[:, -1:] * 0.85
        rolloff_bins = np.argmax(cum_energy >= total_energy, axis=1)
        rolloff_freqs = freqs[rolloff_bins]
        mean_rolloff = float(np.mean(rolloff_freqs))

        return {
            'spectral_centroid_hz': round(mean_centroid, 2),
            'spectral_flatness': round(mean_flatness, 4),
            'spectral_rolloff_hz': round(mean_rolloff, 2)
        }

    # Method aliases for flexibility
    extract_mfcc_with_deltas = extract_mfcc
    extract_pitch_f0_autocorr = estimate_pitch_f0
    extract_spectral_features = compute_spectral_moments
