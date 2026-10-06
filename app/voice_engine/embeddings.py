import hashlib
import json
import logging
import math
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _length_to_mask(
    length: torch.Tensor,
    max_len: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Creates a binary mask for each sequence of length `length`."""
    if max_len is None:
        max_len = int(length.max().item())
    if device is None:
        device = length.device
    if dtype is None:
        dtype = length.dtype
    mask = torch.arange(max_len, device=device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return torch.as_tensor(mask, dtype=dtype, device=device)


class _Conv1d(nn.Module):
    """Pure-PyTorch 1D convolution matching SpeechBrain ECAPA-TDNN state_dict and reflection padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.dilation * (self.kernel_size - 1)) // 2
        if pad > 0:
            x = F.pad(x, (pad, pad), mode="reflect")
        return self.conv(x)


class _BatchNorm1d(nn.Module):
    """Pure-PyTorch 1D BatchNorm wrapper matching SpeechBrain ECAPA-TDNN state_dict."""

    def __init__(self, input_size: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TDNNBlock(nn.Module):
    """Time-Delay Neural Network (TDNN) block with dilated reflection-padded Conv1d, ReLU, and BatchNorm1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: type[nn.Module] = nn.ReLU,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = _Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = _BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.activation(self.conv(x))))


class Res2NetBlock(nn.Module):
    """Multi-scale Res2Net block with hierarchical dilated TDNN sub-branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        y_i = None
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = _Conv1d(in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _Conv1d(in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        L = x.shape[-1]
        if lengths is not None:
            mask = _length_to_mask(lengths * L, max_len=L, device=x.device).unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """Channel- and context-dependent Attentive Statistics Pooling (ASP)."""

    def __init__(self, channels: int, attention_channels: int = 128, global_context: bool = True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        in_ch = channels * 3 if global_context else channels
        self.tdnn = TDNNBlock(in_ch, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = _Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        L = x.shape[-1]

        def _compute_statistics(t: torch.Tensor, m: torch.Tensor, dim: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
            mean = (m * t).sum(dim)
            std = torch.sqrt((m * (t - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        mask = _length_to_mask(lengths * L, max_len=L, device=x.device).unsqueeze(1)

        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class SERes2NetBlock(nn.Module):
    """Core ECAPA-TDNN building block: TDNN -> Res2Net -> TDNN -> SEBlock with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
            dropout=dropout,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
            dropout=dropout,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)
        self.shortcut = (
            _Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual


class ECAPA_TDNN(nn.Module):
    """Pure-PyTorch implementation of SpeechBrain's ECAPA-TDNN speaker embedding architecture.
    100% state_dict compatible with `speechbrain/spkrec-ecapa-voxceleb` (`embedding_model.ckpt`)
    without requiring `torchaudio` C++ extensions.
    """

    def __init__(
        self,
        input_size: int = 80,
        lin_neurons: int = 192,
        activation: type[nn.Module] = nn.ReLU,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        global_context: bool = True,
        groups: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = channels or [1024, 1024, 1024, 1024, 3072]
        kernel_sizes = kernel_sizes or [5, 3, 3, 3, 1]
        dilations = dilations or [1, 2, 3, 4, 1]
        groups = groups or [1, 1, 1, 1, 1]

        self.channels = channels
        self.blocks = nn.ModuleList()

        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
                dropout,
            )
        )

        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                    dropout=dropout,
                )
            )

        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
            dropout=dropout,
        )

        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = _BatchNorm1d(input_size=channels[-1] * 2)
        self.fc = _Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            if isinstance(layer, TDNNBlock):
                x = layer(x)
            else:
                x = layer(x, lengths=lengths)
            xl.append(x)

        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        x = self.fc(x)
        return x.transpose(1, 2)


class Fbank(nn.Module):
    """Pure-PyTorch 80-bin log-Mel filterbank extractor matching SpeechBrain's Fbank pipeline."""

    def __init__(
        self,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: float = 25.0,
        hop_length: float = 10.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = float(f_max if f_max is not None else sample_rate // 2)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = int(round((sample_rate / 1000.0) * win_length))
        self.hop_length = int(round((sample_rate / 1000.0) * hop_length))
        self.register_buffer("window", torch.hamming_window(self.win_length), persistent=False)

        n_stft = self.n_fft // 2 + 1
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)
        band = hz[1:] - hz[:-1]
        f_central = hz[1:-1]
        all_freqs = torch.linspace(0, self.sample_rate // 2, n_stft)
        all_freqs_mat = all_freqs.repeat(f_central.shape[0], 1)
        f_central_mat = f_central.repeat(all_freqs_mat.shape[1], 1).transpose(0, 1)
        band_mat = band[:-1].repeat(all_freqs_mat.shape[1], 1).transpose(0, 1)

        slope = (all_freqs_mat - f_central_mat) / band_mat
        fbank_matrix = torch.max(
            torch.zeros(1), torch.min(slope + 1.0, -slope + 1.0)
        ).transpose(0, 1)
        self.register_buffer("fbank_matrix", fbank_matrix, persistent=False)

    @staticmethod
    def _to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(wav.device),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        mag = torch.view_as_real(stft).transpose(2, 1).pow(2).sum(-1)
        fbanks = torch.matmul(mag, self.fbank_matrix.to(mag.device))
        x_db = 10.0 * torch.log10(torch.clamp(fbanks, min=1e-10))
        new_x_db_max = x_db.amax(dim=(-2, -1)) - 80.0
        return torch.max(x_db, new_x_db_max.view(x_db.shape[0], 1, 1))


class InputNormalization(nn.Module):
    """Pure-PyTorch sentence-level cepstral mean normalization matching SpeechBrain's InputNormalization."""

    def __init__(self, norm_type: str = "sentence", std_norm: bool = False, epsilon: float = 1e-10):
        super().__init__()
        self.norm_type = norm_type
        self.std_norm = std_norm
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            mean = x.mean(dim=1, keepdim=True)
            if not self.std_norm:
                return x - mean
            std = x.std(dim=1, keepdim=True).clamp(min=self.epsilon)
            return (x - mean) / std

        T = x.shape[1]
        mask = _length_to_mask(lengths * T, max_len=T, device=x.device).unsqueeze(-1)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / n
        if not self.std_norm:
            return x - mean
        var = ((x - mean) * mask).square().sum(dim=1, keepdim=True) / n
        return (x - mean) / var.sqrt().clamp(min=self.epsilon)


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
    HF_CHECKPOINT_URL = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt"

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
                model_dir.mkdir(parents=True, exist_ok=True)
                resp = requests.get(cls.HF_CHECKPOINT_URL, timeout=60)
                resp.raise_for_status()
                ckpt_path.write_bytes(resp.content)
                logger.info("Downloaded ECAPA-TDNN checkpoint from HuggingFace to %s", ckpt_path)
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
                state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                embedding_model.load_state_dict(state_dict, strict=True)
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
