from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from app.models import BiometricAuditLog, VoiceprintProfile
from app.views.auth_views import generate_dynamic_challenge


@login_required
def dashboard_view(request):
    """Main voice authentication dashboard — enrollment, verification, audit."""
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
    """Voiceprint Biometric Enrollment Studio & Vault Settings — redirects to dashboard."""
    user = request.user
    voice_profile, _ = VoiceprintProfile.objects.get_or_create(user=user)

    if request.method == 'POST' and 'update_security_level' in request.POST:
        new_level = request.POST.get('security_level', 'HIGH')
        if new_level in ['STANDARD', 'HIGH', 'STRICT']:
            user.security_level = new_level
            user.save(update_fields=['security_level'])
            messages.success(request, f"Security level updated to {user.get_security_level_display()}.")
            return redirect('app:banking_dashboard')

    # Voice vault now lives on the dashboard — redirect there
    return redirect('app:banking_dashboard')
