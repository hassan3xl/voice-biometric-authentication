import time
import numpy as np
from .preprocessor import AudioPreprocessor
from .features import AcousticFeatureExtractor
from .embeddings import SpeakerEmbeddingEngine
from .anti_spoofing import AntiSpoofingEngine


class VoiceBiometricVerifier:
    """Unified Voice Biometric Verification and Enrollment Pipeline
    for Digital Banking Security.
    """

    # Banking Security Risk Tiers and Decision Thresholds
    RISK_THRESHOLDS = {
        'LOGIN': 0.72,                  # Standard web login
        'STEP_UP_TRANSACTION': 0.80,    # Regular transfers ($500 - $5,000)
        'HIGH_VALUE_WIRE': 0.85,        # High-value wire transfers (> $5,000)
        'SECURITY_SETTINGS': 0.88,      # Re-enrollment or limit elevation
    }

    @classmethod
    def process_audio_pipeline(cls, audio_bytes: bytes) -> dict:
        """Executes full acoustic signal processing pipeline on raw audio bytes.
        Returns:
            dict containing raw audio metrics, preprocessed signal, frames, MFCCs,
            deep embedding, pitch dynamics, and spectral moments.
        """
        t_start = time.perf_counter()

        # 1. Decode & Resample to 16kHz Mono
        raw_signal, sr = AudioPreprocessor.load_audio_from_bytes(audio_bytes)
        raw_duration = len(raw_signal) / sr

        # 2. Compute SNR and Audio Quality
        snr_info = AudioPreprocessor.compute_snr_and_quality(raw_signal, sr=sr)

        # 3. Voice Activity Detection (VAD)
        voiced_signal, speech_duration = AudioPreprocessor.apply_vad(raw_signal, sr=sr)

        # 4. Pre-emphasis
        emphasized = AudioPreprocessor.apply_pre_emphasis(voiced_signal)

        # 5. Hamming Window Framing
        frames = AudioPreprocessor.frame_signal(emphasized, sr=sr)

        # 6. MFCC Feature Extraction (60 dimensions)
        mfcc_matrix = AcousticFeatureExtractor.extract_mfcc(frames, sr=sr)

        # 7. Deep Neural Speaker Embedding (256 dimensions)
        embed_input = voiced_signal if len(voiced_signal) > int(sr * 0.1) else raw_signal
        embedding = SpeakerEmbeddingEngine.extract_embedding(embed_input, sr=sr)

        # 8. Pitch & Harmonic Tracking (F0, Jitter, Shimmer)
        pitch_info = AcousticFeatureExtractor.estimate_pitch_f0(voiced_signal, sr=sr)

        # 9. Spectral Moments (Centroid, Rolloff, Flatness)
        spectral_info = AcousticFeatureExtractor.compute_spectral_moments(frames, sr=sr)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            'raw_duration_sec': round(raw_duration, 2),
            'speech_duration_sec': round(speech_duration, 2),
            'snr_info': snr_info,
            'embedding': embedding,
            'pitch_info': pitch_info,
            'spectral_info': spectral_info,
            'frames_count': frames.shape[0],
            'latency_ms': round(latency_ms, 2),
            'voiced_signal': voiced_signal,
            'signal': voiced_signal if len(voiced_signal) > 0 else raw_signal,
            'sr': sr
        }

    @classmethod
    def verify_speaker(
        cls,
        audio_bytes: bytes,
        enrolled_embedding: np.ndarray | list,
        operation_tier: str = 'STEP_UP_TRANSACTION',
        expected_passphrase: str = "",
        spoken_transcript: str = "",
        custom_threshold: float | None = None
    ) -> dict:
        """Verifies an incoming voice recording against an enrolled master voiceprint.
        
        Evaluates:
        1. Signal Quality & SNR Check
        2. Cosine Similarity & Match Confidence against enrolled template
        3. Anti-Spoofing & Liveness Detection (ISO/IEC 30107)
        4. Risk-Adjusted Decision Boundary
        """
        t_start = time.perf_counter()

        # Step 1: Execute Acoustic Signal Pipeline
        try:
            pipeline_result = cls.process_audio_pipeline(audio_bytes)
        except Exception as e:
            return {
                'decision': 'ERROR',
                'is_authenticated': False,
                'rejection_reason': f"Audio processing error: {str(e)}",
                'latency_ms': round((time.perf_counter() - t_start) * 1000.0, 2),
            }

        snr_info = pipeline_result['snr_info']
        if not snr_info['is_acceptable'] or pipeline_result['speech_duration_sec'] < 0.35:
            return {
                'decision': 'POOR_AUDIO',
                'is_authenticated': False,
                'similarity_score': 0.0,
                'confidence_pct': 0.0,
                'threshold_used': custom_threshold or cls.RISK_THRESHOLDS.get(operation_tier, 0.75),
                'snr_db': snr_info['snr_db'],
                'liveness_score': 0.0,
                'rejection_reason': f"Unusable audio: {snr_info['reason']}. Please speak the phrase clearly into your microphone.",
                'latency_ms': round((time.perf_counter() - t_start) * 1000.0, 2),
            }

        # Step 2: Extract & Compare Embedding Vector
        test_embedding = pipeline_result['embedding']
        similarity_score = SpeakerEmbeddingEngine.compute_cosine_similarity(test_embedding, enrolled_embedding)
        
        # Step 3: Determine Threshold & Calibrate Match Confidence
        threshold = custom_threshold if custom_threshold is not None else cls.RISK_THRESHOLDS.get(operation_tier, 0.75)
        confidence_pct = SpeakerEmbeddingEngine.compute_confidence_percentage(similarity_score, threshold=threshold)
        is_biometric_match = (similarity_score >= threshold)

        # Step 4: Presentation Attack & Anti-Spoofing Verification
        liveness_result = AntiSpoofingEngine.evaluate_liveness(
            signal_data=pipeline_result['voiced_signal'],
            pitch_info=pipeline_result['pitch_info'],
            spectral_info=pipeline_result['spectral_info'],
            snr_info=snr_info,
            expected_passphrase=expected_passphrase,
            spoken_transcript=spoken_transcript
        )

        is_live = liveness_result['is_live']

        # Step 5: Composite Banking Decision Logic
        if not is_live:
            decision = 'SPOOF_DETECTED'
            is_authenticated = False
            rejection_reason = f"Security Violation: {liveness_result['decision_reason']}"
        elif not is_biometric_match:
            decision = 'REJECTED'
            is_authenticated = False
            rejection_reason = f"Voiceprint mismatch (Similarity {round(similarity_score, 3)} < Threshold {round(threshold, 3)})"
        else:
            decision = 'ACCEPTED'
            is_authenticated = True
            rejection_reason = ""

        total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            'decision': decision,
            'is_authenticated': is_authenticated,
            'similarity_score': round(similarity_score, 4),
            'threshold_used': round(threshold, 4),
            'confidence_pct': confidence_pct,
            'liveness_score': liveness_result['liveness_score'],
            'attack_type': liveness_result['attack_type'],
            'is_live': is_live,
            'rejection_reason': rejection_reason,
            'snr_db': snr_info['snr_db'],
            'speech_duration_sec': pipeline_result['speech_duration_sec'],
            'latency_ms': total_latency_ms,
            'diagnostics': {
                'liveness_diagnostics': liveness_result['diagnostics'],
                'f0_mean_hz': pipeline_result['pitch_info']['mean_f0_hz'],
                'spectral_centroid_hz': pipeline_result['spectral_info']['spectral_centroid_hz'],
            }
        }
