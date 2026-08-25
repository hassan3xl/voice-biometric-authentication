from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from app.models import BiometricAuditLog, VoiceprintProfile
from app.views.auth_views import generate_dynamic_challenge


@login_required
def dashboard_view(request):
    """Lightweight voice authentication dashboard."""
    user = request.user
    voice_profile, _ = VoiceprintProfile.objects.get_or_create(user=user)
    recent_audit_logs = BiometricAuditLog.objects.filter(user=user).order_by('-created_at')[:5]
    challenge_phrase = generate_dynamic_challenge(user)

    context = {
        'user': user,
        'voice_profile': voice_profile,
        'recent_audit_logs': recent_audit_logs,
        'challenge_phrase': challenge_phrase,
        'verification_count': recent_audit_logs.count(),
    }
    return render(request, 'app/dashboard.html', context)


@login_required
def voice_vault_view(request):
    """Voiceprint Biometric Enrollment Studio & Vault Settings."""
    user = request.user
    voice_profile, _ = VoiceprintProfile.objects.get_or_create(user=user)

    if request.method == 'POST' and 'update_security_level' in request.POST:
        new_level = request.POST.get('security_level', 'HIGH')
        if new_level in ['STANDARD', 'HIGH', 'STRICT']:
            user.security_level = new_level
            user.save(update_fields=['security_level'])
            messages.success(request, f"Voice biometric security level updated to {user.get_security_level_display()}.")
            return redirect('app:voice_vault')

    # Passphrase sequence for 3-pass enrollment
    enrollment_passes = [
        {'pass_num': 1, 'title': 'Master Security Passphrase', 'phrase': 'My voice is my secure key for voice biometri'},
        {'pass_num': 2, 'title': 'Numeric Cadence Verification', 'phrase': 'Authorize biometric voice access for user profile'},
        {'pass_num': 3, 'title': 'Acoustic Liveness Confirmation', 'phrase': 'Voice Biometri identity verified and sealed'},
    ]

    recent_samples = voice_profile.samples.all()

    return render(request, 'app/voice_vault.html', {
        'user': user,
        'voice_profile': voice_profile,
        'enrollment_passes': enrollment_passes,
        'recent_samples': recent_samples,
    })


def _seed_demo_banking_data(user):
    """Seeds initial sample transactions and beneficiaries for realistic banking experience."""
    beneficiaries_data = [
        {'name': 'Sophia Laurent (CloudTech Inc)', 'account_number': '4091823741', 'bank_name': 'Apex Federal Bank', 'email': 'laurent@cloudtech.io'},
        {'name': 'Marcus Vance', 'account_number': '4019284729', 'bank_name': 'Chase Manhattan', 'email': 'marcus.v@gmail.com'},
        {'name': 'Apex High-Yield Treasury Fund', 'account_number': '4088219034', 'bank_name': 'Apex Federal Bank', 'email': 'treasury@apexbank.com'},
    ]
    for b in beneficiaries_data:
        Beneficiary.objects.create(user=user, **b)

    sample_txns = [
        {'recipient_name': 'CloudTech Enterprise SaaS', 'recipient_account': '4091823741', 'amount': Decimal('1450.00'), 'category': 'BILL_PAY', 'description': 'Annual cloud hosting license', 'status': 'COMPLETED', 'step_up_required': True, 'voice_auth_score': 0.88, 'voice_auth_confidence': 96.4},
        {'recipient_name': 'Marcus Vance', 'recipient_account': '4019284729', 'amount': Decimal('450.00'), 'category': 'TRANSFER', 'description': 'Shared advisory retainer', 'status': 'COMPLETED', 'step_up_required': False},
        {'recipient_name': 'Apex High-Yield Treasury Fund', 'recipient_account': '4088219034', 'amount': Decimal('5000.00'), 'category': 'WIRE', 'description': 'Quarterly treasury allocation', 'status': 'COMPLETED', 'step_up_required': True, 'voice_auth_score': 0.91, 'voice_auth_confidence': 98.7},
    ]
    for t in sample_txns:
        Transaction.objects.create(sender=user, completed_at=timezone.now(), **t)
