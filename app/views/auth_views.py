import base64
import random
import time
import numpy as np
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from app.forms.auth_forms import BankingLoginForm, BankingRegistrationForm
from app.models import BiometricAuditLog, VoiceprintProfile
from app.voice_engine.embeddings import SpeakerEmbeddingEngine
from app.voice_engine.anti_spoofing import AntiSpoofingEngine
from app.voice_engine.verifier import VoiceBiometricVerifier

User = get_user_model()

# Pre-defined voice authentication dynamic passphrases
DYNAMIC_PASSPHRASE_TEMPLATES = [
    "My voice is my secure key for voice biometric",
    "Voice Biometric authentication code {code}",
    "Authorize biometric voice session for user {account_suffix}",
    "Voice Biometric acoustic token {code} verified",
    "Confirm secure biometric identity clearance for voice biometric",
]


def generate_dynamic_challenge(user=None) -> str:
    """Generates a time-bound dynamic voice challenge passphrase."""
    code = random.randint(1000, 9999)
    acc_suffix = user.account_number[-4:] if user else f"{random.randint(1000, 9999)}"
    template = random.choice(DYNAMIC_PASSPHRASE_TEMPLATES)
    return template.format(code=code, account_suffix=acc_suffix)


def login_view(request):
    """Renders the login portal with password and voice biometric authentication."""
    if request.user.is_authenticated:
        return redirect(request.GET.get('next') or 'app:banking_dashboard')

    next_url = request.GET.get('next', '')

    if request.method == 'POST':
        form = BankingLoginForm(request.POST)
        if form.is_valid():
            identifier = form.cleaned_data['username_or_email'].strip()
            password = form.cleaned_data['password']

            # Check if identifier is email or account number
            user_obj = None
            if '@' in identifier:
                user_obj = User.objects.filter(email__iexact=identifier).first()
            else:
                user_obj = User.objects.filter(account_number=identifier).first() or User.objects.filter(username__iexact=identifier).first()

            if user_obj:
                user = authenticate(request, username=user_obj.username, password=password)
                if user is not None:
                    login(request, user)
                    messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                    
                    # If user hasn't enrolled voiceprint, suggest voice vault enrollment
                    if not user.voice_enrolled:
                        messages.info(request, "Secure your account: enroll your Voice Biometric Vault for 1-click login.")
                    
                    return redirect(request.POST.get('next') or request.GET.get('next') or 'app:banking_dashboard')

            messages.error(request, "Invalid account credentials. Please verify your email/account number and password.")
    else:
        form = BankingLoginForm()

    # Pre-generate dynamic challenge for voice login modal
    initial_challenge = generate_dynamic_challenge()

    return render(request, 'auth/login.html', {
        'form': form,
        'initial_challenge': initial_challenge,
        'next_url': next_url,
    })


def register_view(request):
    """Customer onboarding for the voice authentication module."""
    if request.user.is_authenticated:
        return redirect('app:banking_dashboard')

    if request.method == 'POST':
        form = BankingRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Initialize empty voiceprint profile
            VoiceprintProfile.objects.create(user=user, status='PENDING')
            login(request, user)
            messages.success(request, f"Account #{user.account_number} created successfully!")
            messages.info(request, "Please proceed to enroll your Voice Biometric Vault to activate voice login.")
            return redirect('app:voice_vault')
        else:
            messages.error(request, "Please correct the registration errors below.")
    else:
        form = BankingRegistrationForm()

    return render(request, 'auth/register.html', {'form': form})


def logout_view(request):
    """Secure session termination."""
    logout(request)
    messages.info(request, "You have been securely logged out of Apex Vault.")
    return redirect('app:login')


def get_voice_challenge_api(request):
    """API returning a dynamic challenge passphrase for voice authentication."""
    user = request.user if request.user.is_authenticated else None
    challenge = generate_dynamic_challenge(user)
    return JsonResponse({
        'status': 'success',
        'challenge_phrase': challenge,
        'timestamp': time.time(),
    })


@csrf_exempt
@require_POST
def voice_login_api(request):
    """Voice login API that identifies the account owner from the voice sample itself."""
    audio_base64 = request.POST.get('audio_data', '')
    challenge_phrase = request.POST.get('challenge_phrase', '')
    spoken_transcript = request.POST.get('spoken_transcript', '')

    if not audio_base64:
        return JsonResponse({'status': 'error', 'message': 'Voice audio stream was not captured.'}, status=400)

    # Extract audio bytes
    try:
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Failed to decode audio payload: {str(e)}'}, status=400)

    # Run Voice Biometric Verification Pipeline once, then compare against all enrolled users
    client_ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', '127.0.0.1')).split(',')[0].strip()
    user_agent = request.META.get('HTTP_USER_AGENT', '')

    try:
        pipeline = VoiceBiometricVerifier.process_audio_pipeline(audio_bytes)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Audio processing error: {str(e)}'}, status=400)

    # Check audio quality and speech presence
    snr_info = pipeline['snr_info']
    if not snr_info.get('is_acceptable', False) or pipeline.get('speech_duration_sec', 0.0) < 0.35:
        reason = snr_info.get('reason', 'No speech detected')
        audit_log = BiometricAuditLog.objects.create(
            user=None,
            attempt_type='LOGIN',
            challenge_phrase=challenge_phrase,
            spoken_transcript=spoken_transcript,
            similarity_score=0.0,
            threshold_used=0.72,
            confidence_score=0.0,
            liveness_score=0.0,
            snr_db=snr_info.get('snr_db', 0.0),
            decision='POOR_AUDIO',
            rejection_reason=f"Acoustic gate failed: {reason}",
            attack_type='NONE',
            latency_ms=pipeline.get('latency_ms', 0.0),
            client_ip=client_ip,
            user_agent=user_agent,
        )
        return JsonResponse({
            'status': 'rejected',
            'authenticated': False,
            'decision': 'POOR_AUDIO',
            'message': f"{reason}. Please speak the phrase aloud into your microphone.",
            'metrics': {
                'similarity_score': 0.0,
                'confidence_pct': 0.0,
                'liveness_score': 0.0,
                'latency_ms': pipeline.get('latency_ms', 0.0),
                'attack_type': 'NONE',
                'audit_id': audit_log.id,
            }
        }, status=400)

    liveness_result = AntiSpoofingEngine.evaluate_liveness(
        signal_data=pipeline['voiced_signal'],
        pitch_info=pipeline['pitch_info'],
        spectral_info=pipeline['spectral_info'],
        snr_info=pipeline['snr_info'],
        expected_passphrase=challenge_phrase,
        spoken_transcript=spoken_transcript,
    )

    if not liveness_result['is_live']:
        audit_log = BiometricAuditLog.objects.create(
            user=None,
            attempt_type='LOGIN',
            challenge_phrase=challenge_phrase,
            spoken_transcript=spoken_transcript,
            similarity_score=0.0,
            threshold_used=0.72,
            confidence_score=0.0,
            liveness_score=liveness_result.get('liveness_score', 0.0),
            snr_db=pipeline['snr_info']['snr_db'],
            decision='SPOOF_DETECTED',
            rejection_reason=liveness_result.get('decision_reason', 'Voice liveness check failed.'),
            attack_type=liveness_result.get('attack_type', 'NONE'),
            latency_ms=pipeline.get('latency_ms', 0.0),
            client_ip=client_ip,
            user_agent=user_agent,
        )
        return JsonResponse({
            'status': 'rejected',
            'authenticated': False,
            'decision': 'SPOOF_DETECTED',
            'message': liveness_result.get('decision_reason', 'Voice liveness check failed.'),
            'metrics': {
                'similarity_score': 0.0,
                'confidence_pct': 0.0,
                'liveness_score': liveness_result.get('liveness_score', 0.0),
                'latency_ms': pipeline.get('latency_ms', 0.0),
                'attack_type': liveness_result.get('attack_type', 'NONE'),
                'audit_id': audit_log.id,
            }
        }, status=401)

    test_embedding = pipeline['embedding']
    enrolled_profiles = VoiceprintProfile.objects.filter(status='ACTIVE').select_related('user')
    best_profile = None
    best_similarity = -1.0

    for profile in enrolled_profiles:
        if not profile.embedding_vector:
            continue
        similarity = SpeakerEmbeddingEngine.compute_cosine_similarity(test_embedding, profile.embedding_vector)
        if similarity > best_similarity:
          best_similarity = similarity
          best_profile = profile

    threshold = 0.72
    if best_profile is None or best_similarity < threshold:
        audit_log = BiometricAuditLog.objects.create(
            user=None,
            attempt_type='LOGIN',
            challenge_phrase=challenge_phrase,
            spoken_transcript=spoken_transcript,
            similarity_score=max(0.0, best_similarity),
            threshold_used=threshold,
            confidence_score=SpeakerEmbeddingEngine.compute_confidence_percentage(max(0.0, best_similarity), threshold=threshold),
            liveness_score=liveness_result.get('liveness_score', 0.0),
            snr_db=pipeline['snr_info']['snr_db'],
            decision='REJECTED',
            rejection_reason='No enrolled voice profile matched the speaker.',
            attack_type=liveness_result.get('attack_type', 'NONE'),
            latency_ms=pipeline.get('latency_ms', 0.0),
            client_ip=client_ip,
            user_agent=user_agent,
        )
        return JsonResponse({
            'status': 'rejected',
            'authenticated': False,
            'decision': 'REJECTED',
            'message': 'No enrolled voice profile matched the speaker.',
            'metrics': {
                'similarity_score': max(0.0, best_similarity),
                'confidence_pct': SpeakerEmbeddingEngine.compute_confidence_percentage(max(0.0, best_similarity), threshold=threshold),
                'liveness_score': liveness_result.get('liveness_score', 0.0),
                'latency_ms': pipeline.get('latency_ms', 0.0),
                'attack_type': liveness_result.get('attack_type', 'NONE'),
                'audit_id': audit_log.id,
            }
        }, status=401)

    user = best_profile.user
    verify_result = {
        'is_authenticated': True,
        'similarity_score': best_similarity,
        'confidence_pct': SpeakerEmbeddingEngine.compute_confidence_percentage(best_similarity, threshold=threshold),
        'liveness_score': liveness_result.get('liveness_score', 0.0),
        'attack_type': liveness_result.get('attack_type', 'NONE'),
        'latency_ms': pipeline.get('latency_ms', 0.0),
        'snr_db': pipeline['snr_info']['snr_db'],
        'decision': 'ACCEPTED',
        'rejection_reason': '',
    }

    # Create Biometric Forensic Audit Log
    audit_log = BiometricAuditLog.objects.create(
        user=user,
        attempt_type='LOGIN',
        challenge_phrase=challenge_phrase,
        spoken_transcript=spoken_transcript,
        similarity_score=verify_result.get('similarity_score', 0.0),
        threshold_used=threshold,
        confidence_score=verify_result.get('confidence_pct', 0.0),
        liveness_score=verify_result.get('liveness_score', 0.0),
        snr_db=verify_result.get('snr_db', 0.0),
        decision=verify_result.get('decision', 'REJECTED'),
        rejection_reason=verify_result.get('rejection_reason', ''),
        attack_type=verify_result.get('attack_type', 'NONE'),
        latency_ms=verify_result.get('latency_ms', 0.0),
        client_ip=client_ip,
        user_agent=user_agent
    )

    if verify_result['is_authenticated']:
        # Biometric Authentication Succeeded! Log customer in
        login(request, user)
        best_profile.last_verified_at = timezone.now()
        best_profile.save(update_fields=['last_verified_at'])

        return JsonResponse({
            'status': 'success',
            'authenticated': True,
            'decision': 'ACCEPTED',
            'message': f'Voice Biometric Authentication Verified! Welcome, {user.first_name or user.username}.',
            'redirect_url': '/dashboard/',
            'metrics': {
                'similarity_score': verify_result['similarity_score'],
                'confidence_pct': verify_result['confidence_pct'],
                'liveness_score': verify_result['liveness_score'],
                'latency_ms': verify_result['latency_ms'],
                'snr_db': verify_result['snr_db'],
                'audit_id': audit_log.id,
            }
        })
    else:
        return JsonResponse({
            'status': 'rejected',
            'authenticated': False,
            'decision': verify_result.get('decision'),
            'message': verify_result.get('rejection_reason', 'Voice biometric verification failed.'),
            'metrics': {
                'similarity_score': verify_result.get('similarity_score', 0.0),
                'confidence_pct': verify_result.get('confidence_pct', 0.0),
                'liveness_score': verify_result.get('liveness_score', 0.0),
                'latency_ms': verify_result.get('latency_ms', 0.0),
                'attack_type': verify_result.get('attack_type', 'NONE'),
            }
        }, status=401)
