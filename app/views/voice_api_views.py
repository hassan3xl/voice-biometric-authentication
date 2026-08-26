import base64
import time
from decimal import Decimal
from django.contrib.auth.decorators import login_required
from django.db import transaction as db_transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from app.models import BiometricAuditLog, Transaction, VoiceEnrollmentSample, VoiceprintProfile
from app.voice_engine.embeddings import SpeakerEmbeddingEngine
from app.voice_engine.verifier import VoiceBiometricVerifier


@login_required
@require_POST
def voice_enroll_api(request):  
    """Multi-Pass Voiceprint Enrollment API.
    Captures acoustic sample, computes 192D embedding, and on final pass,
    fuses master template with cryptographic voiceprint hash.
    """
    user = request.user
    voice_profile, _ = VoiceprintProfile.objects.get_or_create(user=user)

    pass_index = int(request.POST.get('pass_index', 1))
    audio_base64 = request.POST.get('audio_data', '')
    passphrase = request.POST.get('passphrase', 'My voice is my secure key for Apex Bank')

    if not audio_base64:
        return JsonResponse({'status': 'error', 'message': 'Audio data not provided.'}, status=400)

    try:
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid audio payload: {str(e)}'}, status=400)

    # Execute acoustic signal processing pipeline
    try:
        pipeline = VoiceBiometricVerifier.process_audio_pipeline(audio_bytes)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Acoustic processing error: {str(e)}'}, status=500)

    snr_info = pipeline['snr_info']
    if not snr_info['is_acceptable'] or pipeline['speech_duration_sec'] < 0.35:
        return JsonResponse({
            'status': 'error',
            'quality_issue': True,
            'message': f"Acoustic quality rejected: {snr_info['reason']}. Please speak the phrase aloud into your microphone.",
            'snr_db': snr_info['snr_db'],
        }, status=400)

    sample_embedding = pipeline['embedding']

    # Delete existing sample at this index if re-recording pass
    voice_profile.samples.filter(sample_index=pass_index).delete()

    # Save enrollment sample
    VoiceEnrollmentSample.objects.create(
        profile=voice_profile,
        sample_index=pass_index,
        passphrase=passphrase,
        audio_duration=pipeline['speech_duration_sec'],
        snr_db=snr_info['snr_db'],
        embedding_json=sample_embedding.tolist()
    )

    all_samples = voice_profile.samples.all()
    sample_count = all_samples.count()

    is_enrollment_complete = False
    master_hash = ""
    intra_variance = 0.0

    if sample_count >= 3:
        # Fuse all passes into master voiceprint template
        embeddings_list = [s.embedding_json for s in all_samples]
        master_embedding, intra_variance, master_hash = SpeakerEmbeddingEngine.fuse_enrollment_samples(embeddings_list)

        # Update Voiceprint Profile
        voice_profile.embedding_vector = master_embedding.tolist()
        voice_profile.sample_count = sample_count
        voice_profile.intra_variance_score = intra_variance
        voice_profile.voiceprint_hash = master_hash
        voice_profile.baseline_f0_hz = pipeline['pitch_info']['mean_f0_hz']
        voice_profile.baseline_jitter = pipeline['pitch_info']['jitter_pct']
        voice_profile.baseline_spectral_centroid = pipeline['spectral_info']['spectral_centroid_hz']
        voice_profile.status = 'ACTIVE'
        voice_profile.save()

        # Update User
        user.voice_enrolled = True
        user.voice_enrolled_at = timezone.now()
        user.save(update_fields=['voice_enrolled', 'voice_enrolled_at'])

        # Create Biometric Audit Log
        BiometricAuditLog.objects.create(
            user=user,
            attempt_type='RE_ENROLLMENT',
            challenge_phrase=passphrase,
            similarity_score=1.0,
            confidence_score=99.9,
            liveness_score=1.0,
            snr_db=snr_info['snr_db'],
            decision='ACCEPTED',
            latency_ms=pipeline['latency_ms'],
            client_ip=request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
        is_enrollment_complete = True

    return JsonResponse({
        'status': 'success',
        'pass_index': pass_index,
        'sample_count': sample_count,
        'is_complete': is_enrollment_complete,
        'snr_db': snr_info['snr_db'],
        'duration_sec': pipeline['speech_duration_sec'],
        'pitch_hz': pipeline['pitch_info']['mean_f0_hz'],
        'intra_variance': intra_variance,
        'voiceprint_hash': master_hash,
        'message': 'Voiceprint enrollment complete! Your voice vault is active.' if is_enrollment_complete else f'Pass {pass_index} captured successfully.'
    })


@login_required
@require_POST
def voice_verify_step_up_api(request):
    """Step-Up Voice Biometric Verification API for High-Value Wire & Banking Transactions."""
    user = request.user
    transaction_id = request.POST.get('transaction_id')
    audio_base64 = request.POST.get('audio_data', '')
    challenge_phrase = request.POST.get('challenge_phrase', '')
    spoken_transcript = request.POST.get('spoken_transcript', '')

    txn = get_object_or_404(Transaction, transaction_id=transaction_id, sender=user)

    if txn.status == 'COMPLETED':
        return JsonResponse({'status': 'success', 'message': 'Transaction already approved and completed.'})

    voice_profile = getattr(user, 'voice_profile', None)
    if not voice_profile or voice_profile.status != 'ACTIVE' or not voice_profile.embedding_vector:
        return JsonResponse({
            'status': 'error',
            'message': 'Voice Vault is not enrolled on your account. Please complete enrollment first.'
        }, status=400)

    try:
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid audio payload: {str(e)}'}, status=400)

    tier = 'HIGH_VALUE_WIRE' if txn.amount >= Decimal('5000.00') else 'STEP_UP_TRANSACTION'

    verify_result = VoiceBiometricVerifier.verify_speaker(
        audio_bytes=audio_bytes,
        enrolled_embedding=voice_profile.embedding_vector,
        operation_tier=tier,
        expected_passphrase=challenge_phrase or txn.challenge_phrase_used,
        spoken_transcript=spoken_transcript,
    )

    client_ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', '127.0.0.1')).split(',')[0].strip()

    # Log forensic biometric audit attempt
    audit_log = BiometricAuditLog.objects.create(
        user=user,
        attempt_type='STEP_UP_TRANSACTION',
        challenge_phrase=challenge_phrase or txn.challenge_phrase_used,
        spoken_transcript=spoken_transcript,
        similarity_score=verify_result.get('similarity_score', 0.0),
        threshold_used=verify_result.get('threshold_used', 0.80),
        confidence_score=verify_result.get('confidence_pct', 0.0),
        liveness_score=verify_result.get('liveness_score', 0.0),
        snr_db=verify_result.get('snr_db', 0.0),
        decision=verify_result.get('decision', 'REJECTED'),
        rejection_reason=verify_result.get('rejection_reason', ''),
        attack_type=verify_result.get('attack_type', 'NONE'),
        latency_ms=verify_result.get('latency_ms', 0.0),
        client_ip=client_ip,
        user_agent=request.META.get('HTTP_USER_AGENT', '')
    )

    if verify_result['is_authenticated']:
        # Succeeded! Atomically deduct balance and complete transaction
        with db_transaction.atomic():
            if user.checking_balance < txn.amount:
                return JsonResponse({'status': 'error', 'message': 'Insufficient funds in checking account.'}, status=400)

            user.checking_balance -= txn.amount
            user.save(update_fields=['checking_balance'])

            txn.status = 'COMPLETED'
            txn.voice_auth_score = verify_result['similarity_score']
            txn.voice_auth_confidence = verify_result['confidence_pct']
            txn.completed_at = timezone.now()
            txn.save()

            voice_profile.last_verified_at = timezone.now()
            voice_profile.save(update_fields=['last_verified_at'])

        return JsonResponse({
            'status': 'success',
            'authenticated': True,
            'message': f'Voice Biometric Authorization Confirmed! ${txn.amount:,.2f} wire transfer executed.',
            'redirect_url': f'/transaction/{txn.transaction_id}/',
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
            'message': verify_result.get('rejection_reason', 'Voice authorization failed.'),
            'metrics': {
                'similarity_score': verify_result.get('similarity_score', 0.0),
                'confidence_pct': verify_result.get('confidence_pct', 0.0),
                'liveness_score': verify_result.get('liveness_score', 0.0),
                'latency_ms': verify_result.get('latency_ms', 0.0),
                'attack_type': verify_result.get('attack_type', 'NONE'),
            }
        }, status=401)


@login_required
@require_POST
def voice_mic_test_api(request):
    """Microphone Audio Diagnostics API for Real-Time Feedback."""
    audio_base64 = request.POST.get('audio_data', '')
    if not audio_base64:
        return JsonResponse({'status': 'error', 'message': 'No audio data received.'}, status=400)

    try:
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_base64)
        pipeline = VoiceBiometricVerifier.process_audio_pipeline(audio_bytes)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({
        'status': 'success',
        'snr_db': pipeline['snr_info']['snr_db'],
        'clipping_pct': pipeline['snr_info']['clipping_pct'],
        'is_acceptable': pipeline['snr_info']['is_acceptable'],
        'duration_sec': pipeline['speech_duration_sec'],
        'pitch_hz': pipeline['pitch_info']['mean_f0_hz'],
        'spectral_centroid_hz': pipeline['spectral_info']['spectral_centroid_hz'],
        'latency_ms': pipeline['latency_ms'],
    })


@login_required
@require_POST
def voice_reset_api(request):
    """Revokes current voiceprint and resets enrollment."""
    user = request.user
    voice_profile, _ = VoiceprintProfile.objects.get_or_create(user=user)
    
    # Remove samples and reset profile
    voice_profile.samples.all().delete()
    voice_profile.embedding_vector = []
    voice_profile.sample_count = 0
    voice_profile.intra_variance_score = 0.0
    voice_profile.voiceprint_hash = ""
    voice_profile.status = 'PENDING'
    voice_profile.save()

    user.voice_enrolled = False
    user.save(update_fields=['voice_enrolled'])

    return JsonResponse({'status': 'success', 'message': 'Voiceprint Vault reset successfully. You can now re-enroll.'})


@login_required
@require_POST
def voice_test_verify_api(request):
    """Direct Voice Biometric Verification Test API for Dashboard Demonstration."""
    user = request.user
    audio_base64 = request.POST.get('audio_data', '')
    challenge_phrase = request.POST.get('challenge_phrase', '')
    spoken_transcript = request.POST.get('spoken_transcript', '')

    voice_profile = getattr(user, 'voice_profile', None)
    if not voice_profile or voice_profile.status != 'ACTIVE' or not voice_profile.embedding_vector:
        return JsonResponse({
            'status': 'error',
            'message': 'Voice Vault is not enrolled on your account. Please complete voice enrollment first.'
        }, status=400)

    try:
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid audio payload: {str(e)}'}, status=400)

    verify_result = VoiceBiometricVerifier.verify_speaker(
        audio_bytes=audio_bytes,
        enrolled_embedding=voice_profile.embedding_vector,
        operation_tier='LOGIN',
        expected_passphrase=challenge_phrase,
        spoken_transcript=spoken_transcript,
    )

    client_ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', '127.0.0.1')).split(',')[0].strip()

    audit_log = BiometricAuditLog.objects.create(
        user=user,
        attempt_type='LOGIN',
        challenge_phrase=challenge_phrase,
        spoken_transcript=spoken_transcript,
        similarity_score=verify_result.get('similarity_score', 0.0),
        threshold_used=verify_result.get('threshold_used', 0.72),
        confidence_score=verify_result.get('confidence_pct', 0.0),
        liveness_score=verify_result.get('liveness_score', 0.0),
        snr_db=verify_result.get('snr_db', 0.0),
        decision=verify_result.get('decision', 'REJECTED'),
        rejection_reason=verify_result.get('rejection_reason', ''),
        attack_type=verify_result.get('attack_type', 'NONE'),
        latency_ms=verify_result.get('latency_ms', 0.0),
        client_ip=client_ip,
        user_agent=request.META.get('HTTP_USER_AGENT', '')
    )

    if verify_result['is_authenticated']:
        voice_profile.last_verified_at = timezone.now()
        voice_profile.save(update_fields=['last_verified_at'])
        return JsonResponse({
            'status': 'success',
            'authenticated': True,
            'decision': 'ACCEPTED',
            'message': 'Voice Biometric Verification Successful! Identity matched with enrolled template.',
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
            'decision': verify_result.get('decision', 'REJECTED'),
            'message': verify_result.get('rejection_reason', 'Voice verification failed.'),
            'metrics': {
                'similarity_score': verify_result.get('similarity_score', 0.0),
                'confidence_pct': verify_result.get('confidence_pct', 0.0),
                'liveness_score': verify_result.get('liveness_score', 0.0),
                'latency_ms': verify_result.get('latency_ms', 0.0),
                'snr_db': verify_result.get('snr_db', 0.0),
                'attack_type': verify_result.get('attack_type', 'NONE'),
                'audit_id': audit_log.id,
            }
        }, status=401)

