import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import torch

# Ensure torchaudio loads cleanly on CPU-only environments even if PyPI wheel built with CUDA symbols is present
try:
    import torchaudio._extension.utils as _ta_utils

    _orig_ta_load = _ta_utils._load_lib

    def _safe_ta_load(lib: str) -> bool:
        try:
            return _orig_ta_load(lib)
        except OSError:
            return False

    _ta_utils._load_lib = _safe_ta_load
except Exception:
    pass

from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.processing.features import InputNormalization

logger = logging.getLogger(__name__)


class SpeakerEmbeddingEngine:
    """Deep Neural Acoustic Speaker Recognition & Verification Engine (ECAPA-TDNN).

    Extracts 192-dimensional speaker embeddings combining SpeechBrain's pretrained
    ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
    with vocal-tract spectral & glottal harmonic invariants.
    Provides high-precision speaker discrimination, cosine similarity scoring,
    and multi-pass intra-speaker enrollment fusion.
    """

    EMBEDDING_DIM = 192
    CALIBRATION_STEEPNESS = 14.0
    CALIBRATION_MIDPOINT = 0.72  # Standard match threshold
    SAMPLE_RATE = 16000
    MEL_CHANNELS = 80

    _compute_features: Fbank | None = None
    _mean_var_norm: InputNormalization | None = None
    _embedding_model: ECAPA_TDNN | None = None
    _ecapa_center: torch.Tensor | None = None
    _mel_filterbank: np.ndarray | None = None

    @classmethod
    def _get_ecapa_modules(cls) -> tuple[Fbank, InputNormalization, ECAPA_TDNN, torch.Tensor]:
        """Loads and returns singleton pretrained SpeechBrain ECAPA-TDNN modules and calibrated center vector."""
        if cls._embedding_model is not None and cls._ecapa_center is not None:
            return cls._compute_features, cls._mean_var_norm, cls._embedding_model, cls._ecapa_center

        model_dir = Path(__file__).resolve().parent / "models" / "spkrec-ecapa-voxceleb"
        ckpt_path = model_dir / "embedding_model.ckpt"

        if not ckpt_path.exists():
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                model_dir.mkdir(parents=True, exist_ok=True)
                EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(model_dir),
                    run_opts={"device": "cpu"},
                )
            except Exception as e:
                logger.warning("Could not auto-download ECAPA-TDNN from HuggingFace: %s", e)

        compute_features = Fbank(n_mels=cls.MEL_CHANNELS)
        mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
        embedding_model = ECAPA_TDNN(
            input_size=cls.MEL_CHANNELS,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=cls.EMBEDDING_DIM,
        )

        if ckpt_path.exists():
            try:
                state_dict = torch.load(str(ckpt_path), map_location="cpu")
                embedding_model.load_state_dict(state_dict)
                logger.info("Loaded pretrained SpeechBrain ECAPA-TDNN (192-D) from %s", ckpt_path)
            except Exception as e:
                logger.error("Failed to load ECAPA-TDNN weights from %s: %s", ckpt_path, e)

        embedding_model.eval()

        # Calibrate isotropic background center across synthetic pitch/formant reference cohort
        # to remove the stationary BatchNorm bias of ECAPA-TDNN at inference
        with torch.no_grad():
            t = torch.linspace(0, 1.2, int(cls.SAMPLE_RATE * 1.2)).unsqueeze(0)
            cohort = []
            for f0 in (95.0, 130.0, 175.0, 230.0, 285.0):
                sig = (
                    0.5 * torch.sin(2 * np.pi * f0 * t)
                    + 0.3 * torch.sin(2 * np.pi * (2.0 * f0) * t)
                    + 0.2 * torch.sin(2 * np.pi * (3.0 * f0) * t)
                ) * (0.6 + 0.4 * torch.sin(2 * np.pi * 3.5 * t))
                cohort.append(sig)
            batch_wav = torch.cat(cohort, dim=0)
            batch_lens = torch.ones(len(cohort), dtype=torch.float32)
            feats = compute_features(batch_wav)
            feats_cmn = mean_var_norm(feats, batch_lens)
            feats_spec = feats - feats.mean(dim=(1, 2), keepdim=True)
            feats_in = 0.4 * feats_cmn + 0.6 * feats_spec
            cohort_embs = embedding_model(feats_in, batch_lens).squeeze(1)
            ecapa_center = cohort_embs.mean(dim=0).detach()

        cls._compute_features = compute_features
        cls._mean_var_norm = mean_var_norm
        cls._embedding_model = embedding_model
        cls._ecapa_center = ecapa_center
        return cls._compute_features, cls._mean_var_norm, cls._embedding_model, cls._ecapa_center

    @classmethod
    def _get_mel_filterbank(cls, sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = MEL_CHANNELS) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix (n_mels, n_fft//2 + 1)."""
        if cls._mel_filterbank is not None and cls._mel_filterbank.shape[0] == n_mels:
            return cls._mel_filterbank

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(60.0)
        mel_max = hz_to_mel(sr / 2.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_prev = bin_points[m - 1]
            f_curr = bin_points[m]
            f_next = bin_points[m + 1]

            for k in range(f_prev, f_curr):
                filterbank[m - 1, k] = (k - f_prev) / max(1, (f_curr - f_prev))
            for k in range(f_curr, f_next):
                filterbank[m - 1, k] = (f_next - k) / max(1, (f_next - f_curr))

        cls._mel_filterbank = filterbank
        return cls._mel_filterbank

    @classmethod
    def compute_mel_spectrogram(cls, signal_data: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Computes 80-channel log-Mel spectrogram from waveform."""
        frame_len = int(sr * 0.025)  # 400 samples (25 ms)
        hop_len = int(sr * 0.010)    # 160 samples (10 ms)
        n_fft = 512

        if len(signal_data) < frame_len:
            signal_data = np.pad(signal_data, (0, frame_len - len(signal_data)))

        window = np.hamming(frame_len)
        num_frames = max(1, (len(signal_data) - frame_len) // hop_len + 1)
        frames = np.zeros((num_frames, frame_len), dtype=np.float32)
        for i in range(num_frames):
            frames[i] = signal_data[i * hop_len : i * hop_len + frame_len] * window

        power_spec = (np.abs(np.fft.rfft(frames, n_fft)) ** 2) / n_fft
        mel_basis = cls._get_mel_filterbank(sr, n_fft, cls.MEL_CHANNELS)
        mel_spec = np.dot(power_spec, mel_basis.T)
        log_mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        return log_mel_spec.astype(np.float32)

    @classmethod
    def _extract_vocal_tract_subspace(cls, log_mel: np.ndarray) -> np.ndarray:
        """Extracts a 96-D zero-mean L2-normalized vocal-tract & glottal harmonic subspace vector.
        Combines 64-D log-Mel vocal-tract spectral envelope with 32-D harmonic cepstrum.
        """
        # 1. Static vocal-tract spectral envelope (80 Mel bands interpolated to 64-D, zero-centered)
        spec_mean = np.mean(log_mel, axis=0)
        idx_64 = np.linspace(0, len(spec_mean) - 1, 64)
        env_64 = np.interp(idx_64, np.arange(len(spec_mean)), spec_mean).astype(np.float32)
        env_64 = env_64 - np.mean(env_64)
        norm_e = float(np.linalg.norm(env_64))
        if norm_e > 1e-9:
            env_64 = env_64 / norm_e

        # 2. Real cepstrum of the log-Mel envelope (32-D glottal pitch & formant spacing profile)
        cepstrum = np.fft.irfft(spec_mean - np.mean(spec_mean), n=64)[:32].astype(np.float32)
        cepstrum = cepstrum - np.mean(cepstrum)
        norm_c = float(np.linalg.norm(cepstrum))
        if norm_c > 1e-9:
            cepstrum = cepstrum / norm_c

        subspace_96 = np.concatenate([env_64 * 0.75, cepstrum * 0.66]).astype(np.float32)
        norm_total = float(np.linalg.norm(subspace_96))
        if norm_total > 1e-9:
            subspace_96 = subspace_96 / norm_total
        return subspace_96

    @classmethod
    def extract_embedding(cls, signal_or_features: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extracts 192-dimensional speaker embedding combining whitened SpeechBrain ECAPA-TDNN
        deep neural representations (96-D subspace) with vocal-tract spectral invariants (96-D subspace).

        Args:
            signal_or_features: 1D waveform (recommended) or 2D acoustic frame matrix.
            sr: Audio sample rate (default: 16,000 Hz).

        Returns:
            192-dimensional L2-normalized numpy float32 array.
        """
        compute_features, mean_var_norm, embedding_model, ecapa_center = cls._get_ecapa_modules()

        if signal_or_features.ndim == 1:
            signal_data = signal_or_features.astype(np.float32)
            if len(signal_data) < int(sr * 0.1):  # < 100ms
                return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

            # Remove DC offset and normalize peak amplitude
            signal_data = signal_data - np.mean(signal_data)
            peak = np.max(np.abs(signal_data))
            if peak > 1e-6:
                signal_data = signal_data / peak

            log_mel_np = cls.compute_mel_spectrogram(signal_data, sr=sr)
            vocal_tract_96 = cls._extract_vocal_tract_subspace(log_mel_np)

            # Wrap-pad short voiced segments (< 1.2s) with active speech frames rather than zeros
            min_samples = int(sr * 1.2)
            if len(signal_data) < min_samples:
                signal_data = np.pad(signal_data, (0, min_samples - len(signal_data)), mode="wrap")

            wav_tensor = torch.from_numpy(signal_data).unsqueeze(0)
            wav_lens = torch.ones(1, dtype=torch.float32)

            with torch.no_grad():
                feats = compute_features(wav_tensor)
                feats_cmn = mean_var_norm(feats, wav_lens)
                feats_spec = feats - feats.mean(dim=(1, 2), keepdim=True)
                feats_in = 0.4 * feats_cmn + 0.6 * feats_spec
                raw_embed = embedding_model(feats_in, wav_lens).squeeze(0).squeeze(0)
        else:
            # 2D feature matrix fallback
            if signal_or_features.shape[0] < 2:
                return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
            mel = signal_or_features.astype(np.float32)
            if mel.shape[1] != cls.MEL_CHANNELS:
                if mel.shape[1] < cls.MEL_CHANNELS:
                    mel = np.pad(mel, ((0, 0), (0, cls.MEL_CHANNELS - mel.shape[1])), mode="edge")
                else:
                    mel = mel[:, : cls.MEL_CHANNELS]
            vocal_tract_96 = cls._extract_vocal_tract_subspace(mel)
            feats_tensor = torch.from_numpy(mel).unsqueeze(0)
            wav_lens = torch.ones(1, dtype=torch.float32)
            with torch.no_grad():
                feats_cmn = mean_var_norm(feats_tensor, wav_lens)
                feats_spec = feats_tensor - feats_tensor.mean(dim=(1, 2), keepdim=True)
                feats_in = 0.4 * feats_cmn + 0.6 * feats_spec
                raw_embed = embedding_model(feats_in, wav_lens).squeeze(0).squeeze(0)

        # Center ECAPA-TDNN embedding against calibrated cohort mean and compress to 96-D neural subspace
        centered_ecapa = (raw_embed - ecapa_center).cpu().numpy().astype(np.float32)
        centered_ecapa = centered_ecapa - np.mean(centered_ecapa)
        ecapa_96 = (centered_ecapa[:96] + centered_ecapa[96:]) * 0.5
        ecapa_norm = float(np.linalg.norm(ecapa_96))
        if ecapa_norm > 1e-9:
            ecapa_96 = ecapa_96 / ecapa_norm

        # Concatenate orthogonal 96-D ECAPA-TDNN subspace and 96-D Vocal-Tract subspace -> 192-D unit vector
        combined_192 = np.concatenate([ecapa_96 * np.sqrt(0.5), vocal_tract_96 * np.sqrt(0.5)]).astype(np.float32)
        total_norm = float(np.linalg.norm(combined_192))
        if total_norm > 1e-9:
            combined_192 = combined_192 / total_norm

        return combined_192

    @classmethod
    def compute_cosine_similarity(cls, emb1: np.ndarray | list, emb2: np.ndarray | list) -> float:
        """Computes Cosine Similarity between two 192-D speaker embeddings:
        CosSim(A, B) = (A . B) / (||A|| * ||B||) in [-1.0, 1.0].
        """
        a = np.array(emb1, dtype=np.float32).flatten()
        b = np.array(emb2, dtype=np.float32).flatten()

        # Reject comparisons between incompatible model dimensions (e.g., legacy 256-D LSTM vs 192-D ECAPA-TDNN)
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    @classmethod
    def compute_confidence_percentage(cls, similarity_score: float, threshold: float = CALIBRATION_MIDPOINT) -> float:
        """Calibrates similarity score into a posterior match confidence probability [0% - 100%].
        Uses Sigmoid Calibration: P(Match) = 1 / (1 + exp(-k * (score - threshold)))
        """
        k = cls.CALIBRATION_STEEPNESS
        prob = 1.0 / (1.0 + np.exp(-k * (similarity_score - threshold)))
        return round(float(prob * 100.0), 2)

    # Aliases for API flexibility
    compute_speaker_embedding = extract_embedding
    calibrate_posterior_confidence = compute_confidence_percentage

    @classmethod
    def fuse_enrollment_samples(cls, embedding_list: list[np.ndarray | list]) -> tuple[np.ndarray, float, str]:
        """Fuses multiple enrollment sample embeddings into a master 192-D voiceprint template.
        Calculates intra-speaker consistency and generates a cryptographic voiceprint hash.
        Returns: (master_embedding, intra_variance_score, voiceprint_hash)
        """
        if not embedding_list:
            raise ValueError("No embeddings provided for enrollment fusion.")

        arrays = [np.array(e, dtype=np.float32).flatten() for e in embedding_list]

        # Ensure all passes have consistent dimensionality
        target_dim = len(arrays[-1])
        arrays = [a for a in arrays if len(a) == target_dim]

        # Calculate pairwise similarities to assess intra-speaker consistency
        n = len(arrays)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = cls.compute_cosine_similarity(arrays[i], arrays[j])
                pairwise_sims.append(sim)

        avg_consistency = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0
        intra_variance = max(0.0, 1.0 - avg_consistency)

        # Centroid fusion: average L2-normalized vectors and re-normalize onto unit hypersphere
        normalized_arrays = [a / (np.linalg.norm(a) + 1e-9) for a in arrays]
        sum_vec = np.sum(normalized_arrays, axis=0)
        norm = np.linalg.norm(sum_vec)
        master_embedding = (sum_vec / norm) if norm > 1e-9 else sum_vec

        # Cryptographic Voiceprint Hash (SHA-256 of quantized vector representation)
        quantized = np.round(master_embedding * 10000).astype(int).tobytes()
        voiceprint_hash = hashlib.sha256(quantized).hexdigest()

        return master_embedding.astype(np.float32), round(intra_variance, 4), voiceprint_hash

    @classmethod
    def serialize_embedding(cls, embedding: np.ndarray) -> str:
        """Serializes numpy embedding vector to JSON string for database storage."""
        return json.dumps(embedding.tolist())

    @classmethod
    def deserialize_embedding(cls, embedding_str: str | list) -> np.ndarray:
        """Deserializes JSON string or list back to numpy float32 embedding."""
        if isinstance(embedding_str, str):
            return np.array(json.loads(embedding_str), dtype=np.float32)
        return np.array(embedding_str, dtype=np.float32)
