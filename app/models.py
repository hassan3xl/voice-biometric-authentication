import random
import uuid
from decimal import Decimal
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


def generate_account_number():
    """Generates unique 10-digit digital banking account number."""
    return f"40{random.randint(10000000, 99999999)}"


class User(AbstractUser):
    """Custom User model representing Digital Banking Customers and Security Administrators."""
    
    SECURITY_LEVEL_CHOICES = (
        ('STANDARD', 'Standard Security (Password + Optional Voice)'),
        ('HIGH', 'High Security (Voice Step-Up on Transfers > $1,000)'),
        ('STRICT', 'Strict Treasury Grade (Voice Required for All Transactions)'),
    )

    RISK_TIER_CHOICES = (
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk'),
    )

    account_number = models.CharField(max_length=12, unique=True, default=generate_account_number)
    phone_number = models.CharField(max_length=20, blank=True, default="+1 (555) 019-2834")
    
    # Financial Balances
    checking_balance = models.DecimalField(max_digits=14, decimal_places=2, default=Decimal('15840.50'))
    savings_balance = models.DecimalField(max_digits=14, decimal_places=2, default=Decimal('42300.00'))
    investment_balance = models.DecimalField(max_digits=14, decimal_places=2, default=Decimal('28750.75'))

    # Voice Biometrics Status
    voice_enrolled = models.BooleanField(default=False)
    voice_enrolled_at = models.DateTimeField(null=True, blank=True)
    voice_login_enabled = models.BooleanField(default=True)
    security_level = models.CharField(max_length=20, choices=SECURITY_LEVEL_CHOICES, default='HIGH')
    risk_tier = models.CharField(max_length=10, choices=RISK_TIER_CHOICES, default='LOW')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def total_balance(self) -> Decimal:
        return self.checking_balance + self.savings_balance + self.investment_balance

    @property
    def formatted_account_number(self) -> str:
        acc = str(self.account_number)
        if len(acc) == 10:
            return f"{acc[:3]} {acc[3:6]} {acc[6:]}"
        return acc

    def __str__(self):
        return f"{self.username} ({self.get_full_name() or self.email}) - Acc #{self.account_number}"


class VoiceprintProfile(models.Model):
    """Cryptographic Voiceprint Biometric Profile for Speaker Verification."""
    
    STATUS_CHOICES = (
        ('ACTIVE', 'Active & Verified'),
        ('PENDING', 'Enrollment Incomplete'),
        ('REVOKED', 'Revoked / Re-enrollment Required'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='voice_profile')
    embedding_vector = models.JSONField(default=list, help_text="192-dimensional L2 normalized acoustic vector")
    enrollment_passphrase = models.CharField(max_length=255, default="My voice is my secure key for voice biometric")
    sample_count = models.IntegerField(default=0)
    intra_variance_score = models.FloatField(default=0.0, help_text="Intra-speaker sample consistency variance")
    
    # Acoustic Baselines
    baseline_f0_hz = models.FloatField(default=125.0)
    baseline_jitter = models.FloatField(default=0.85)
    baseline_spectral_centroid = models.FloatField(default=2250.0)
    
    # Cryptographic integrity signature
    voiceprint_hash = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    last_verified_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Voice Vault: {self.user.username} [{self.status}] (Samples: {self.sample_count})"


class VoiceEnrollmentSample(models.Model):
    """Individual recording pass stored during multi-pass voice enrollment."""
    
    profile = models.ForeignKey(VoiceprintProfile, on_delete=models.CASCADE, related_name='samples')
    sample_index = models.IntegerField(default=1)
    passphrase = models.CharField(max_length=255)
    audio_duration = models.FloatField(default=0.0)
    snr_db = models.FloatField(default=0.0)
    embedding_json = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['sample_index']

    def __str__(self):
        return f"Sample #{self.sample_index} for {self.profile.user.username} (SNR: {self.snr_db}dB)"


class Beneficiary(models.Model):
    """Saved verified transfer recipients for the banking customer."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='beneficiaries')
    name = models.CharField(max_length=150)
    account_number = models.CharField(max_length=20)
    bank_name = models.CharField(max_length=100, default="Apex Federal Bank")
    routing_number = models.CharField(max_length=20, default="021000021")
    email = models.EmailField(blank=True)
    is_verified = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} - {self.bank_name} ({self.account_number})"


class Transaction(models.Model):
    """Banking transactions requiring step-up voice biometric verification."""
    
    CATEGORY_CHOICES = (
        ('TRANSFER', 'Direct Transfer'),
        ('WIRE', 'High-Value Wire Transfer'),
        ('BILL_PAY', 'Bill Payment'),
        ('INSTANT', 'Instant Voice Cash'),
    )

    STATUS_CHOICES = (
        ('COMPLETED', 'Completed'),
        ('PENDING_VOICE_AUTH', 'Pending Voice Authorization'),
        ('REJECTED', 'Biometric Auth Rejected'),
        ('CANCELLED', 'Cancelled'),
    )

    transaction_id = models.CharField(max_length=36, unique=True, default=uuid.uuid4, editable=False)
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_transactions')
    recipient_name = models.CharField(max_length=150)
    recipient_account = models.CharField(max_length=30)
    amount = models.DecimalField(max_digits=14, decimal_places=2)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='TRANSFER')
    description = models.TextField(blank=True)
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='COMPLETED')
    
    # Step-Up Security
    step_up_required = models.BooleanField(default=False)
    voice_auth_score = models.FloatField(null=True, blank=True)
    voice_auth_confidence = models.FloatField(null=True, blank=True)
    challenge_phrase_used = models.CharField(max_length=255, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"TXN-{str(self.transaction_id)[:8]} | ${self.amount} -> {self.recipient_name} [{self.status}]"


class BiometricAuditLog(models.Model):
    """Forensic Biometric Audit Trail for all voice authentication and verification attempts."""
    
    ATTEMPT_TYPE_CHOICES = (
        ('LOGIN', 'Voice Biometric Login'),
        ('STEP_UP_TRANSACTION', 'High-Value Transaction Step-Up'),
        ('BENEFICIARY_ADD', 'Beneficiary Addition Step-Up'),
        ('RE_ENROLLMENT', 'Voiceprint Enrollment / Update'),
        ('TEST_VERIFY', 'Diagnostic Voice Verification'),
    )

    DECISION_CHOICES = (
        ('ACCEPTED', 'Biometric Match Accepted'),
        ('REJECTED', 'Voiceprint Mismatch Rejected'),
        ('SPOOF_DETECTED', 'Presentation Attack Detected'),
        ('POOR_AUDIO', 'Unusable Acoustic Quality'),
        ('ERROR', 'Processing Error'),
    )

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='biometric_logs')
    attempt_type = models.CharField(max_length=30, choices=ATTEMPT_TYPE_CHOICES, default='LOGIN')
    challenge_phrase = models.CharField(max_length=255)
    spoken_transcript = models.CharField(max_length=255, blank=True)
    
    # Biometric Metrics
    similarity_score = models.FloatField(default=0.0)
    threshold_used = models.FloatField(default=0.75)
    confidence_score = models.FloatField(default=0.0)
    liveness_score = models.FloatField(default=0.0)
    snr_db = models.FloatField(default=0.0)
    
    decision = models.CharField(max_length=20, choices=DECISION_CHOICES, default='ACCEPTED')
    rejection_reason = models.CharField(max_length=255, blank=True)
    attack_type = models.CharField(max_length=50, default='NONE')
    
    # Performance & Device
    latency_ms = models.FloatField(default=0.0)
    client_ip = models.CharField(max_length=45, default='127.0.0.1')
    user_agent = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        usr = self.user.username if self.user else "Anonymous"
        return f"[{self.created_at.strftime('%H:%M:%S')}] {self.attempt_type} - {usr} -> {self.decision} (Sim: {self.similarity_score})"


class BiometricBenchmarkRun(models.Model):
    """Historical performance evaluation runs stored for compliance and research reporting."""
    
    run_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    total_trials = models.IntegerField(default=0)
    genuine_trials = models.IntegerField(default=0)
    impostor_trials = models.IntegerField(default=0)
    spoof_trials = models.IntegerField(default=0)
    
    # Accuracy Metrics
    eer_score = models.FloatField(default=0.0)
    eer_threshold = models.FloatField(default=0.75)
    auc_roc = models.FloatField(default=0.0)
    
    # ISO/IEC 30107 PAD Metrics
    apcer = models.FloatField(default=0.0)
    bpcer = models.FloatField(default=0.0)
    acer = models.FloatField(default=0.0)
    
    avg_latency_ms = models.FloatField(default=0.0)
    details_json = models.JSONField(default=dict)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Benchmark #{str(self.run_id)[:8]} - EER: {self.eer_score}% | AUC: {self.auc_roc}"
