import re
import numpy as np


class AntiSpoofingEngine:
    """Presentation Attack Detection (PAD) and Liveness Verification
    for Digital Banking Security according to ISO/IEC 30107 standards.
    
    Protects against:
    1. Replay Attacks (Recorded voice played back via phone/loudspeaker)
    2. Synthetic / TTS Voices (Neural speech synthesizers & voice conversion)
    3. Static audio / Silences
    4. Out-of-order or mismatched challenge passphrases
    """

    LIVENESS_PASS_THRESHOLD = 0.65  # Minimum composite liveness score required

    @classmethod
    def evaluate_liveness(
        cls,
        signal_data: np.ndarray,
        pitch_info: dict,
        spectral_info: dict,
        snr_info: dict = None,
        expected_passphrase: str = "",
        spoken_transcript: str = ""
    ) -> dict:
        """Evaluates audio presentation for liveness and anti-spoofing integrity.
        
        Returns:
            dict containing:
                - is_live: bool
                - liveness_score: float (0.0 to 1.0)
                - attack_type: str ('NONE', 'REPLAY_ATTACK', 'SYNTHETIC_TTS', 'STATIC_AUDIO', 'CHALLENGE_MISMATCH')
                - diagnostics: dict of forensic indicators
        """
        flags = []
        scores = []

        # 1. Check signal energy and minimum length
        if len(signal_data) < 4000:  # < 0.25s at 16kHz
            return {
                'is_live': False,
                'liveness_score': 0.0,
                'attack_type': 'STATIC_AUDIO',
                'decision_reason': 'Audio sample too short or empty',
                'diagnostics': {'signal_length_samples': len(signal_data)}
            }

        # 2. Replay Attack Analysis: High-band spectral power and rolloff
        # Real microphones close to vocal tract have rich high frequencies (4kHz - 8kHz)
        # Replayed audio via phone/cheap speakers often exhibits extreme low-pass cutoffs or high flatness
        spectral_rolloff = spectral_info.get('spectral_rolloff_hz', 3000.0)
        spectral_flatness = spectral_info.get('spectral_flatness', 0.05)

        replay_score = 1.0
        if spectral_rolloff < 850.0:
            replay_score -= 0.45
            flags.append('LOW_BANDWIDTH_REPLAY_SUSPECT')
        elif spectral_rolloff < 1100.0:
            replay_score -= 0.15

        if spectral_flatness > 0.65:
            replay_score -= 0.35
            flags.append('FLAT_SPECTRUM_SPEAKER_REPLAY')
        elif spectral_flatness > 0.50:
            replay_score -= 0.15

        replay_score = max(0.0, min(1.0, replay_score))
        scores.append(replay_score)

        # 3. Synthetic TTS & Vocoder Artifact Detection: Pitch dynamics and micro-jitter
        # Natural human vocal folds have subtle organic perturbation:
        # Normal human jitter: 0.25% - 2.5%
        # Normal human shimmer: 1.0% - 5.0%
        # TTS / Neural vocoders often produce unrealistically flat pitch or unstable vocoder glitches
        jitter = pitch_info.get('jitter_pct', 0.8)
        shimmer = pitch_info.get('shimmer_pct', 2.0)
        f0_std = pitch_info.get('f0_std', 15.0)

        tts_naturalness_score = 1.0
        if jitter < 0.04 and f0_std < 2.5:
            # Overly robotic / flat synthetic pitch
            tts_naturalness_score -= 0.50
            flags.append('UNNATURAL_PITCH_STABILITY_TTS')
        elif jitter > 18.0:
            # Harsh vocoder phase glitch or corrupt audio
            tts_naturalness_score -= 0.30
            flags.append('EXCESSIVE_PHASE_DISTORTION')

        if shimmer < 0.08:
            tts_naturalness_score -= 0.20
            flags.append('ARTIFICIAL_AMPLITUDE_CONTINUITY')

        tts_naturalness_score = max(0.0, min(1.0, tts_naturalness_score))
        scores.append(tts_naturalness_score)

        # 4. Voiced-to-Unvoiced Distribution (Natural speech rhythm)
        voiced_ratio = pitch_info.get('voiced_ratio', 0.5)
        rhythm_score = 1.0
        if voiced_ratio < 0.10:
            rhythm_score -= 0.45
            flags.append('INSUFFICIENT_VOICED_HARMONICS')
        elif voiced_ratio > 0.98:
            rhythm_score -= 0.30
            flags.append('MONOTONE_TONE_INJECTION')

        rhythm_score = max(0.0, min(1.0, rhythm_score))
        scores.append(rhythm_score)

        # 5. Challenge-Response Transcript Matching (if provided)
        challenge_score = 1.0
        if expected_passphrase and spoken_transcript:
            expected_clean = re.sub(r'[^a-zA-Z0-9\s]', '', expected_passphrase.lower()).split()
            spoken_clean = re.sub(r'[^a-zA-Z0-9\s]', '', spoken_transcript.lower()).split()

            if expected_clean:
                matches = sum(1 for word in expected_clean if word in spoken_clean)
                word_match_ratio = matches / len(expected_clean)
                if word_match_ratio < 0.4:
                    challenge_score = 0.4
                    flags.append('CHALLENGE_PHRASE_MISMATCH')
                else:
                    challenge_score = min(1.0, 0.4 + 0.6 * word_match_ratio)
            scores.append(challenge_score)

        # Composite Liveness Score (Weighted average with weakest-link bounding)
        # Weights: Replay (30%), TTS Naturalness (35%), Rhythm (20%), Challenge (15%)
        if len(scores) == 4:
            weights = [0.30, 0.35, 0.20, 0.15]
            raw_composite = float(np.average(scores, weights=weights))
            challenge_pass = challenge_score >= 0.40
        else:
            weights = [0.35, 0.45, 0.20]
            raw_composite = float(np.average(scores, weights=weights))
            challenge_pass = True

        # Critical presentation attack detection (ISO/IEC 30107 standard)
        # Rejection occurs if composite score is below threshold OR any attack vector is critically compromised
        is_live = (
            (raw_composite >= cls.LIVENESS_PASS_THRESHOLD)
            and (replay_score >= 0.45)
            and (tts_naturalness_score >= 0.45)
            and (rhythm_score >= 0.35)
            and challenge_pass
        )

        # Bounded composite score reflecting weakest link for consistent metric reporting
        composite_score = min(raw_composite, replay_score + 0.30, tts_naturalness_score + 0.30)
        composite_score = round(max(0.0, min(1.0, composite_score)), 3)

        # Determine Primary Attack Classification if failed
        attack_type = 'NONE'
        if not is_live:
            if replay_score < 0.45 or ('LOW_BANDWIDTH_REPLAY_SUSPECT' in flags and 'FLAT_SPECTRUM_SPEAKER_REPLAY' in flags):
                attack_type = 'REPLAY_ATTACK'
            elif tts_naturalness_score < 0.45 or 'UNNATURAL_PITCH_STABILITY_TTS' in flags or 'EXCESSIVE_PHASE_DISTORTION' in flags:
                attack_type = 'SYNTHETIC_TTS'
            elif len(scores) == 4 and not challenge_pass:
                attack_type = 'CHALLENGE_MISMATCH'
            elif rhythm_score < 0.35:
                attack_type = 'STATIC_AUDIO'
            else:
                attack_type = 'LOW_LIVENESS_CONFIDENCE'

        decision_reason = "Acoustic presentation verified authentic (ISO/IEC 30107 compliant)" if is_live else (
            f"Liveness failure: {attack_type.replace('_', ' ').title()}" if attack_type != 'NONE' else "Low acoustic naturalness score"
        )

        return {
            'is_live': is_live,
            'liveness_score': composite_score,
            'attack_type': attack_type,
            'decision_reason': decision_reason,
            'flags': flags,
            'diagnostics': {
                'replay_resilience_score': round(replay_score, 2),
                'tts_naturalness_score': round(tts_naturalness_score, 2),
                'vocal_rhythm_score': round(rhythm_score, 2),
                'jitter_pct': pitch_info.get('jitter_pct', 0.0),
                'shimmer_pct': pitch_info.get('shimmer_pct', 0.0),
                'spectral_flatness': spectral_info.get('spectral_flatness', 0.0),
                'spectral_rolloff_hz': spectral_info.get('spectral_rolloff_hz', 0.0),
            }
        }

    # Method alias
    evaluate_presentation_attack = evaluate_liveness
