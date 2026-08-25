import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class _SpeakerEncoderNN(nn.Module):
    """3-layer LSTM + Dense Projection Deep Neural Speaker Encoder (GE2E Architecture)."""

    def __init__(self, mel_n_channels: int = 40, model_hidden_size: int = 256, model_embedding_size: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, 3, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels: torch.FloatTensor) -> torch.FloatTensor:
        """Computes 256-D L2-normalized embeddings for a batch of mel spectrogram frames."""
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        norm = torch.norm(embeds_raw, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)
        return embeds_raw / norm


class SpeakerEmbeddingEngine:
    """Deep Neural Acoustic Speaker Recognition & Verification Engine.
    
    Extracts 256-dimensional neural d-vector speaker embeddings using a deep recurrent
    network trained on thousands of hours of multi-speaker corpora (VoxCeleb).
    Provides high-accuracy speaker discrimination, cosine similarity scoring,
    and intra-speaker enrollment fusion.
    """

    EMBEDDING_DIM = 256
    CALIBRATION_STEEPNESS = 14.0
    CALIBRATION_MIDPOINT = 0.72  # Standard match threshold
    SAMPLE_RATE = 16000
    MEL_CHANNELS = 40

    _model: _SpeakerEncoderNN | None = None
    _mel_filterbank: np.ndarray | None = None

    @classmethod
    def _get_model(cls) -> _SpeakerEncoderNN:
        """Loads and returns singleton pretrained Deep Neural Speaker Encoder."""
        if cls._model is not None:
            return cls._model

        model = _SpeakerEncoderNN(cls.MEL_CHANNELS, cls.EMBEDDING_DIM, cls.EMBEDDING_DIM)

        # Locate pretrained model weights
        weights_paths = [
            Path(__file__).resolve().parent / "models" / "pretrained_speaker_encoder.pt",
            Path(__file__).resolve().parent.parent.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "resemblyzer" / "pretrained.pt",
        ]

        loaded = False
        for p in weights_paths:
            if p.exists():
                try:
                    checkpoint = torch.load(str(p), map_location="cpu")
                    state_dict = {
                        k: v for k, v in checkpoint["model_state"].items()
                        if not k.startswith("similarity_")
                    }
                    model.load_state_dict(state_dict)
                    model.eval()
                    loaded = True
                    logger.info("Loaded pretrained deep speaker encoder from %s", p)
                    break
                except Exception as e:
                    logger.warning("Failed to load weights from %s: %s", p, e)

        if not loaded:
            logger.warning("Pretrained weights not found; initializing randomized encoder weights")
            model.eval()

        cls._model = model
        return cls._model

    @classmethod
    def _get_mel_filterbank(cls, sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = MEL_CHANNELS) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix (n_mels, n_fft//2 + 1)."""
        if cls._mel_filterbank is not None:
            return cls._mel_filterbank

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(0.0)
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
        """Computes linear 40-channel Mel spectrogram from waveform."""
        frame_len = int(sr * 0.025)  # 400 samples (25 ms)
        hop_len = int(sr * 0.010)    # 160 samples (10 ms)
        n_fft = 512

        if len(signal_data) < frame_len:
            signal_data = np.pad(signal_data, (0, frame_len - len(signal_data)))

        window = np.hanning(frame_len)
        num_frames = max(1, (len(signal_data) - frame_len) // hop_len + 1)
        frames = np.zeros((num_frames, frame_len), dtype=np.float32)
        for i in range(num_frames):
            frames[i] = signal_data[i * hop_len : i * hop_len + frame_len] * window

        mag_spec = np.abs(np.fft.rfft(frames, n_fft))
        mel_basis = cls._get_mel_filterbank(sr, n_fft, cls.MEL_CHANNELS)
        mel_spec = np.dot(mag_spec, mel_basis.T)
        return mel_spec.astype(np.float32)

    @classmethod
    def extract_embedding(cls, signal_or_features: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extracts 256-dimensional neural speaker embedding from raw audio signal or acoustic frames.
        
        Args:
            signal_or_features: 1D waveform or 2D feature matrix.
            sr: Audio sample rate (default: 16,000 Hz).
        
        Returns:
            256-dimensional L2-normalized numpy float32 array.
        """
        if signal_or_features.ndim == 1:
            signal_data = signal_or_features.astype(np.float32)
            if len(signal_data) < int(sr * 0.1):  # < 100ms
                return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
            mel = cls.compute_mel_spectrogram(signal_data, sr=sr)
        else:
            # 2D frame matrix
            if signal_or_features.shape[0] < 2:
                return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
            if signal_or_features.shape[1] == cls.MEL_CHANNELS:
                mel = signal_or_features.astype(np.float32)
            else:
                # Interpolate/project features to 40 mel channels
                n_frames = signal_or_features.shape[0]
                mel = np.pad(signal_or_features, ((0, 0), (0, max(0, cls.MEL_CHANNELS - signal_or_features.shape[1]))))[:, :cls.MEL_CHANNELS].astype(np.float32)

        model = cls._get_model()

        # Split utterance into 1.6s partial slices (160 mel frames) with 0.8s hop for temporal pooling
        partial_frames = 160
        hop_frames = 80
        n_mel_frames = len(mel)

        slices = []
        if n_mel_frames <= partial_frames:
            padded = np.pad(mel, ((0, partial_frames - n_mel_frames), (0, 0)))
            slices.append(padded)
        else:
            for start in range(0, n_mel_frames - partial_frames + 1, hop_frames):
                slices.append(mel[start : start + partial_frames])
            # Include final tail slice if needed
            if (n_mel_frames - partial_frames) % hop_frames != 0:
                slices.append(mel[-partial_frames:])

        mels_tensor = torch.from_numpy(np.array(slices, dtype=np.float32))

        with torch.no_grad():
            partial_embeddings = model(mels_tensor).cpu().numpy()

        # Utterance embedding is the centroid of partial embeddings, hyperspherically L2-normalized
        raw_embed = np.mean(partial_embeddings, axis=0)
        norm = np.linalg.norm(raw_embed)
        if norm > 0:
            embedding = raw_embed / norm
        else:
            embedding = raw_embed

        return embedding.astype(np.float32)

    @classmethod
    def compute_cosine_similarity(cls, emb1: np.ndarray | list, emb2: np.ndarray | list) -> float:
        """Computes Cosine Similarity between two speaker embeddings:
        CosSim(A, B) = (A . B) / (||A|| * ||B||) in range [-1.0, 1.0].
        """
        a = np.array(emb1, dtype=np.float32).flatten()
        b = np.array(emb2, dtype=np.float32).flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Handle dimensional compatibility if older 192D embeddings exist
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len] / (np.linalg.norm(a[:min_len]) + 1e-9)
            b = b[:min_len] / (np.linalg.norm(b[:min_len]) + 1e-9)
            norm_a = 1.0
            norm_b = 1.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    @classmethod
    def compute_confidence_percentage(cls, similarity_score: float, threshold: float = CALIBRATION_MIDPOINT) -> float:
        """Calibrates raw cosine similarity into a posterior match confidence probability [0% - 100%].
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
        """Fuses multiple enrollment sample embeddings into a master voiceprint template.
        Calculates intra-speaker consistency and generates a cryptographic voiceprint hash.
        Returns: (master_embedding, intra_variance_score, voiceprint_hash)
        """
        if not embedding_list:
            raise ValueError("No embeddings provided for enrollment fusion.")

        arrays = [np.array(e, dtype=np.float32).flatten() for e in embedding_list]

        # Calculate pairwise cosine similarities to assess consistency
        n = len(arrays)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = cls.compute_cosine_similarity(arrays[i], arrays[j])
                pairwise_sims.append(sim)

        avg_consistency = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0
        intra_variance = max(0.0, 1.0 - avg_consistency)

        # Centroid fusion: average vectors and L2 normalize
        sum_vec = np.sum(arrays, axis=0)
        norm = np.linalg.norm(sum_vec)
        master_embedding = (sum_vec / norm) if norm > 0 else sum_vec

        # Cryptographic Voiceprint Hash (SHA-256 of quantised vector representation)
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

