import base64
import io
import struct
import wave
import numpy as np
from decimal import Decimal
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.utils import timezone

from app.models import (
    VoiceprintProfile,
    VoiceEnrollmentSample,
    Beneficiary,
    Transaction,
    BiometricAuditLog,
    BiometricBenchmarkRun,
)
from app.voice_engine.preprocessor import AudioPreprocessor
from app.voice_engine.features import AcousticFeatureExtractor
from app.voice_engine.embeddings import SpeakerEmbeddingEngine
from app.voice_engine.anti_spoofing import AntiSpoofingEngine
from app.voice_engine.verifier import VoiceBiometricVerifier
from app.voice_engine.evaluator import BiometricEvaluator

User = get_user_model()


def generate_synthetic_wav_bytes(duration_sec=1.5, sample_rate=16000, freq_hz=140.0) -> bytes:
    """Generates synthetic 16kHz 16-bit PCM WAV bytes for unit testing."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # Composite harmonic wave with formant envelope
    audio = 0.6 * np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(2 * np.pi * (freq_hz * 2) * t)
    audio += 0.02 * np.random.randn(num_samples)
    audio = np.clip(audio, -1.0, 1.0)
    
    pcm16 = (audio * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    
    return buf.getvalue()


class VoiceBiometricSignalProcessingTests(TestCase):
    """Tests for Preprocessing, VAD, and Acoustic Feature Extraction."""

    def test_audio_preprocessor_loading_and_vad(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.2, freq_hz=150.0)
        signal, sr = AudioPreprocessor.load_and_resample(wav_bytes)
        
        self.assertEqual(sr, 16000)
        self.assertGreater(len(signal), 10000)
        
        speech_frames, speech_duration = AudioPreprocessor.voice_activity_detection(signal, sr=16000)
        self.assertGreater(len(speech_frames), 0)
        self.assertGreater(speech_duration, 0.0)
        
        snr_info = AudioPreprocessor.compute_signal_to_noise_ratio(signal, sr=16000)
        self.assertIn('snr_db', snr_info)
        self.assertIn('is_acceptable', snr_info)

    def test_mfcc_feature_extraction(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0, freq_hz=160.0)
        signal, sr = AudioPreprocessor.load_and_resample(wav_bytes)
        
        mfcc_60d = AcousticFeatureExtractor.extract_mfcc_with_deltas(signal, sr=16000, n_mfcc=20)
        # 60 dimensions = 20 static + 20 delta + 20 delta-delta
        self.assertEqual(mfcc_60d.shape[1], 60)
        self.assertGreater(mfcc_60d.shape[0], 10)

    def test_pitch_and_spectral_analysis(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0, freq_hz=135.0)
        signal, sr = AudioPreprocessor.load_and_resample(wav_bytes)
        
        pitch_info = AcousticFeatureExtractor.extract_pitch_f0_autocorr(signal, sr=16000)
        self.assertGreater(pitch_info['mean_f0_hz'], 80.0)
        self.assertLess(pitch_info['mean_f0_hz'], 250.0)
        
        spectral_info = AcousticFeatureExtractor.extract_spectral_features(signal, sr=16000)
        self.assertGreater(spectral_info['spectral_centroid_hz'], 0.0)


class SpeakerEmbeddingAndVerificationTests(TestCase):
    """Tests for 256-D Deep Neural Speaker Embeddings and Verification Decisions."""

    def test_embedding_generation_and_normalization(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0, freq_hz=140.0)
        signal, sr = AudioPreprocessor.load_and_resample(wav_bytes)
        
        emb = SpeakerEmbeddingEngine.extract_embedding(signal, sr=sr)
        self.assertEqual(emb.shape, (256,))
        
        # Verify L2 norm is approximately 1.0
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_cosine_similarity_and_confidence_mapping(self):
        rng = np.random.RandomState(42)
        emb1 = rng.randn(256).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        
        # Slight perturbation of emb1 (Genuine match)
        emb2 = emb1 + 0.02 * rng.randn(256).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        
        sim = SpeakerEmbeddingEngine.compute_cosine_similarity(emb1, emb2)
        self.assertGreater(sim, 0.80)
        
        conf = SpeakerEmbeddingEngine.calibrate_posterior_confidence(sim, threshold=0.72)
        self.assertGreater(conf, 75.0)

    def test_multi_pass_sample_fusion(self):
        rng = np.random.RandomState(42)
        samples = []
        for _ in range(3):
            s = rng.randn(256).astype(np.float32)
            s /= np.linalg.norm(s)
            samples.append(s.tolist())
        
        master_emb, variance, sha_hash = SpeakerEmbeddingEngine.fuse_enrollment_samples(samples)
        self.assertEqual(master_emb.shape, (256,))
        self.assertGreater(len(sha_hash), 32)
        self.assertGreaterEqual(variance, 0.0)


class AntiSpoofingAndPADTests(TestCase):
    """Tests for ISO/IEC 30107 Presentation Attack Detection."""

    def test_anti_spoofing_evaluation(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.5, freq_hz=145.0)
        pipeline = VoiceBiometricVerifier.process_audio_pipeline(wav_bytes)
        
        pad_result = AntiSpoofingEngine.evaluate_presentation_attack(
            signal_data=pipeline['signal'],
            pitch_info=pipeline['pitch_info'],
            spectral_info=pipeline['spectral_info'],
            snr_info=pipeline['snr_info'],
            expected_passphrase="My voice is my secure key for Apex Bank",
            spoken_transcript="My voice is my secure key for Apex Bank"
        )
        
        self.assertIn('is_live', pad_result)
        self.assertIn('liveness_score', pad_result)
        self.assertIn('attack_type', pad_result)


class BiometricEvaluatorMetricTests(TestCase):
    """Tests for FAR, FRR, EER, and ISO/IEC 30107 metrics calculation."""

    def test_far_frr_eer_calculation(self):
        rng = np.random.RandomState(42)
        genuine_scores = rng.normal(0.88, 0.05, 300).tolist()
        impostor_scores = rng.normal(0.40, 0.10, 300).tolist()
        
        metrics = BiometricEvaluator.compute_far_frr_eer(genuine_scores, impostor_scores)
        self.assertIn('eer_pct', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertLess(metrics['eer_pct'], 5.0)
        self.assertGreater(metrics['auc_roc'], 0.95)

    def test_iso_pad_metrics(self):
        pad_metrics = BiometricEvaluator.compute_iso_pad_metrics(
            bona_fide_liveness_scores=[0.92, 0.88, 0.95, 0.84, 0.90],
            spoof_liveness_scores=[0.35, 0.42, 0.28, 0.30, 0.40],
            liveness_threshold=0.65
        )
        self.assertEqual(pad_metrics['apcer_pct'], 0.0)
        self.assertEqual(pad_metrics['bpcer_pct'], 0.0)
        self.assertEqual(pad_metrics['acer_pct'], 0.0)


class BankingViewsAndVoiceAPITests(TestCase):
    """Integration tests for Digital Banking Portal, Transfers, and Voice APIs."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='alexander@apexbank.com',
            email='alexander@apexbank.com',
            password='Password123!',
            first_name='Alexander',
            last_name='Hamilton',
            account_number='4091820491',
            checking_balance=Decimal('15000.00'),
            voice_enrolled=True,
            security_level='HIGH'
        )
        
        rng = np.random.RandomState(42)
        base_emb = rng.randn(256).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)
        
        self.profile = VoiceprintProfile.objects.create(
            user=self.user,
            embedding_vector=base_emb.tolist(),
            status='ACTIVE',
            sample_count=3,
            voiceprint_hash='a1b2c3d4e5f67890abcdef1234567890'
        )

    def test_login_page_renders_successfully(self):
        res = self.client.get(reverse('app:login'))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Voice")

    def test_dashboard_authenticated_access(self):
        self.client.login(username='alexander@apexbank.com', password='Password123!')
        res = self.client.get(reverse('app:banking_dashboard'))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Voice Profile")

    def test_voice_challenge_api(self):
        res = self.client.get(reverse('app:api_voice_challenge'))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('challenge_phrase', data)

    def test_voice_login_api_with_audio(self):
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.5, freq_hz=140.0)
        b64_audio = 'data:audio/wav;base64,' + base64.b64encode(wav_bytes).decode('ascii')

        res = self.client.post(reverse('app:api_voice_login'), {
            'audio_data': b64_audio,
            'challenge_phrase': 'Apex Secure Vault verification code 3912',
            'spoken_transcript': 'Apex Secure Vault verification code 3912'
        })
        # Liveness check passes, either logs in or returns similarity verdict
        self.assertIn(res.status_code, [200, 401])
        data = res.json()
        self.assertIn('status', data)
        self.assertIn('decision', data)

    def test_evaluation_dashboard_and_simulation_api(self):
        res = self.client.get(reverse('app:evaluation_dashboard'))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Equal Error Rate")

        sim_res = self.client.get(reverse('app:api_simulate_threshold') + '?threshold=0.80')
        self.assertEqual(sim_res.status_code, 200)
        sim_data = sim_res.json()
        self.assertEqual(sim_data['status'], 'success')
        self.assertIn('far_pct', sim_data['simulation'])
        self.assertIn('frr_pct', sim_data['simulation'])

    def test_voice_enroll_api(self):
        self.client.login(username='alexander@apexbank.com', password='Password123!')
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.2, freq_hz=140.0)
        b64_audio = 'data:audio/wav;base64,' + base64.b64encode(wav_bytes).decode('ascii')
        
        res = self.client.post(reverse('app:api_voice_enroll'), {
            'pass_index': 1,
            'audio_data': b64_audio,
            'passphrase': 'My voice is my secure key for Apex Bank'
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['pass_index'], 1)
