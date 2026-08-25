import numpy as np
from decimal import Decimal
from django.core.management.base import BaseCommand
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
from app.voice_engine.embeddings import SpeakerEmbeddingEngine
from app.voice_engine.evaluator import BiometricEvaluator

User = get_user_model()


class Command(BaseCommand):
    help = 'Seeds initial demonstration data for Apex Voice Biometric Banking system'

    def handle(self, *args, **options):
        self.stdout.write("Seeding Apex Digital Banking demo data...")

        # 1. Primary Demo User: Alexander Hamilton
        alex, created = User.objects.get_or_create(
            username='alexander@apexbank.com',
            defaults={
                'email': 'alexander@apexbank.com',
                'first_name': 'Alexander',
                'last_name': 'Hamilton',
                'phone_number': '+1 (555) 234-8901',
                'account_number': '4091820491',
                'checking_balance': Decimal('18450.75'),
                'savings_balance': Decimal('54200.00'),
                'investment_balance': Decimal('32100.50'),
                'voice_enrolled': True,
                'voice_enrolled_at': timezone.now(),
                'security_level': 'HIGH',
                'risk_tier': 'LOW',
            }
        )
        alex.set_password('Password123!')
        alex.is_staff = True
        alex.is_superuser = True
        alex.save()

        # Seed synthetic acoustic baseline embedding for Alexander
        rng = np.random.RandomState(101)
        base_emb = rng.randn(192).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        alex_profile, _ = VoiceprintProfile.objects.get_or_create(
            user=alex,
            defaults={
                'embedding_vector': base_emb.tolist(),
                'enrollment_passphrase': 'My voice is my secure key for Apex Bank',
                'sample_count': 3,
                'intra_variance_score': 0.042,
                'baseline_f0_hz': 128.4,
                'baseline_jitter': 0.78,
                'baseline_spectral_centroid': 2340.0,
                'voiceprint_hash': '7e89ab4f2c90e318d1a498b2f8190c12e873210459a12bc90fa412356789abcd',
                'status': 'ACTIVE',
                'last_verified_at': timezone.now(),
            }
        )
        if not alex_profile.embedding_vector:
            alex_profile.embedding_vector = base_emb.tolist()
            alex_profile.status = 'ACTIVE'
            alex_profile.sample_count = 3
            alex_profile.save()

        # Seed Beneficiaries
        beneficiaries_data = [
            {'name': 'Sophia Laurent (CloudTech Inc)', 'account_number': '4091823741', 'bank_name': 'Apex Federal Bank', 'email': 'laurent@cloudtech.io'},
            {'name': 'Marcus Vance (Vance Capital)', 'account_number': '4019284729', 'bank_name': 'Chase Manhattan', 'email': 'marcus.v@gmail.com'},
            {'name': 'Apex High-Yield Treasury Fund', 'account_number': '4088219034', 'bank_name': 'Apex Federal Bank', 'email': 'treasury@apexbank.com'},
            {'name': 'Elena Rostova', 'account_number': '4077123984', 'bank_name': 'Barclays International', 'email': 'e.rostova@nexus.uk'},
        ]
        for b in beneficiaries_data:
            Beneficiary.objects.get_or_create(user=alex, account_number=b['account_number'], defaults=b)

        # Seed Recent Transactions
        sample_txns = [
            {'recipient_name': 'CloudTech Enterprise SaaS', 'recipient_account': '4091823741', 'amount': Decimal('1450.00'), 'category': 'BILL_PAY', 'description': 'Annual secure cloud server cluster', 'status': 'COMPLETED', 'step_up_required': True, 'voice_auth_score': 0.89, 'voice_auth_confidence': 97.4},
            {'recipient_name': 'Marcus Vance (Vance Capital)', 'recipient_account': '4019284729', 'amount': Decimal('450.00'), 'category': 'TRANSFER', 'description': 'Monthly advisory retainer', 'status': 'COMPLETED', 'step_up_required': False},
            {'recipient_name': 'Apex High-Yield Treasury Fund', 'recipient_account': '4088219034', 'amount': Decimal('5000.00'), 'category': 'WIRE', 'description': 'Quarterly treasury fixed bond allocation', 'status': 'COMPLETED', 'step_up_required': True, 'voice_auth_score': 0.92, 'voice_auth_confidence': 99.1},
            {'recipient_name': 'Elena Rostova', 'recipient_account': '4077123984', 'amount': Decimal('12000.00'), 'category': 'WIRE', 'description': 'International consulting escrow', 'status': 'COMPLETED', 'step_up_required': True, 'voice_auth_score': 0.88, 'voice_auth_confidence': 96.2},
        ]
        for t in sample_txns:
            Transaction.objects.get_or_create(
                sender=alex,
                recipient_account=t['recipient_account'],
                description=t['description'],
                defaults={**t, 'completed_at': timezone.now()}
            )

        # Seed Forensic Biometric Audit Logs
        audit_samples = [
            {'attempt_type': 'LOGIN', 'challenge_phrase': 'My voice is my secure key for voice biometri', 'spoken_transcript': 'My voice is my secure key for voice biometri', 'similarity_score': 0.89, 'threshold_used': 0.72, 'confidence_score': 97.8, 'liveness_score': 0.92, 'snr_db': 24.5, 'decision': 'ACCEPTED', 'latency_ms': 17.8, 'client_ip': '192.168.1.104'},
            {'attempt_type': 'STEP_UP_TRANSACTION', 'challenge_phrase': 'Voice Biometri authentication token 8492 verified', 'spoken_transcript': 'Voice Biometri authentication token 8492 verified', 'similarity_score': 0.91, 'threshold_used': 0.80, 'confidence_score': 98.9, 'liveness_score': 0.94, 'snr_db': 26.1, 'decision': 'ACCEPTED', 'latency_ms': 18.2, 'client_ip': '192.168.1.104'},
            {'attempt_type': 'LOGIN', 'challenge_phrase': 'Voice Biometri verification code 3912', 'spoken_transcript': 'Voice Biometri verification code 3912', 'similarity_score': 0.78, 'threshold_used': 0.72, 'confidence_score': 74.2, 'liveness_score': 0.41, 'snr_db': 14.2, 'decision': 'SPOOF_DETECTED', 'attack_type': 'REPLAY_ATTACK', 'rejection_reason': 'Security Violation: Low Bandwidth Replay Suspect (Smartphone loudspeaker detected)', 'latency_ms': 19.4, 'client_ip': '45.134.22.89'},
            {'attempt_type': 'LOGIN', 'challenge_phrase': 'Confirm secure identity clearance for voice biometri', 'spoken_transcript': '', 'similarity_score': 0.44, 'threshold_used': 0.72, 'confidence_score': 1.8, 'liveness_score': 0.86, 'snr_db': 22.0, 'decision': 'REJECTED', 'rejection_reason': 'Voiceprint mismatch (Similarity 0.440 < Threshold 0.720)', 'latency_ms': 16.5, 'client_ip': '192.168.1.104'},
            {'attempt_type': 'STEP_UP_TRANSACTION', 'challenge_phrase': 'Authorize biometric voice session for user 0491', 'spoken_transcript': 'Authorize biometric voice session for user 0491', 'similarity_score': 0.90, 'threshold_used': 0.85, 'confidence_score': 98.4, 'liveness_score': 0.91, 'snr_db': 25.4, 'decision': 'ACCEPTED', 'latency_ms': 18.0, 'client_ip': '192.168.1.104'},
        ]
        for a in audit_samples:
            BiometricAuditLog.objects.create(user=alex, **a)

        # 2. Impostor / Unenrolled Customer: Sarah Jenkins
        sarah, _ = User.objects.get_or_create(
            username='sarah.jenkins@example.com',
            defaults={
                'email': 'sarah.jenkins@example.com',
                'first_name': 'Sarah',
                'last_name': 'Jenkins',
                'account_number': '4082910384',
                'checking_balance': Decimal('6200.00'),
                'savings_balance': Decimal('14500.00'),
                'investment_balance': Decimal('5000.00'),
                'voice_enrolled': False,
                'security_level': 'HIGH',
            }
        )
        sarah.set_password('Password123!')
        sarah.save()
        VoiceprintProfile.objects.get_or_create(user=sarah, defaults={'status': 'PENDING'})

        # 3. Seed Biometric Performance Benchmark Run
        benchmark_data = BiometricEvaluator.generate_empirical_benchmark_dataset(seed=42)
        summary = benchmark_data['summary']
        BiometricBenchmarkRun.objects.create(
            total_trials=summary['total_trials'],
            genuine_trials=summary['genuine_trials'],
            impostor_trials=summary['impostor_trials'],
            spoof_trials=summary['spoof_trials'],
            eer_score=summary['eer_pct'],
            eer_threshold=summary['eer_threshold'],
            auc_roc=summary['auc_roc'],
            apcer=summary['apcer_pct'],
            bpcer=summary['bpcer_pct'],
            acer=summary['acer_pct'],
            avg_latency_ms=18.4,
            details_json=benchmark_data
        )

        self.stdout.write(self.style.SUCCESS("Successfully seeded demo data! User: alexander@apexbank.com | Password: Password123!"))
