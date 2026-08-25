"""Voice Biometric Authentication Engine for Digital Banking Security.
Includes signal preprocessing, MFCC & spectral feature extraction,
deep acoustic embeddings, anti-spoofing / liveness detection,
and statistical decision scoring.
"""

from .preprocessor import AudioPreprocessor
from .features import AcousticFeatureExtractor
from .embeddings import SpeakerEmbeddingEngine
from .anti_spoofing import AntiSpoofingEngine
from .verifier import VoiceBiometricVerifier

__all__ = [
    'AudioPreprocessor',
    'AcousticFeatureExtractor',
    'SpeakerEmbeddingEngine',
    'AntiSpoofingEngine',
    'VoiceBiometricVerifier',
]
