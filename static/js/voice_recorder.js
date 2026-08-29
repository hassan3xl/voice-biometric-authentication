/**
 * Apex Voice Biometric Web Audio Recording Engine.
 * Captures uncompressed 16kHz 16-bit Mono PCM WAV audio directly in browser
 * with linear downsampling, cross-device AudioContext compatibility,
 * and real-time visualizers.
 */

class ApexVoiceRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.analyserNode = null;
        this.processorNode = null;
        this.pcmData = [];
        this.isRecording = false;
        this.targetSampleRate = 16000;
        this.animFrameId = null;
        this.speechRecognizer = null;
        this.transcript = '';
    }

    /**
     * Initializes microphone stream and Web Audio nodes safely across desktop & mobile browsers.
     */
    async init() {
        // 1. Create and resume AudioContext synchronously on user touch/click gesture thread
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            throw new Error("Web Audio API is not supported in this browser.");
        }

        if (!this.audioContext || this.audioContext.state === 'closed') {
            try {
                this.audioContext = new AudioCtx({ sampleRate: this.targetSampleRate });
            } catch (e) {
                this.audioContext = new AudioCtx();
            }
        }

        if (this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
            } catch (e) {
                console.warn("AudioContext resume warning:", e);
            }
        }

        // 2. Clean up previous stream if inactive
        if (this.mediaStream) {
            const tracks = this.mediaStream.getAudioTracks();
            if (tracks.length > 0 && tracks[0].readyState !== 'live') {
                this.mediaStream = null;
            }
        }

        // 3. Request microphone access
        if (!this.mediaStream) {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    const legacyGetUserMedia = navigator.getUserMedia ||
                                               navigator.webkitGetUserMedia ||
                                               navigator.mozGetUserMedia ||
                                               navigator.msGetUserMedia;
                    if (legacyGetUserMedia) {
                        this.mediaStream = await new Promise((resolve, reject) => {
                            legacyGetUserMedia.call(navigator, { audio: true }, resolve, reject);
                        });
                    } else {
                        const isSecure = window.location.protocol === 'https:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
                        if (!isSecure) {
                            throw new Error("Microphone access requires HTTPS. Please access this application via HTTPS (e.g., https://" + window.location.host + ").");
                        } else {
                            throw new Error("Microphone permission or API not available in this browser context.");
                        }
                    }
                } else {
                    this.mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                }
            } catch (err) {
                console.error("Microphone access failed:", err);
                throw new Error(err.message || "Microphone permission denied or not available.");
            }
        }

        // Initialize Web Speech Recognition if available
        if (!this.speechRecognizer && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            try {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                this.speechRecognizer = new SpeechRecognition();
                this.speechRecognizer.continuous = true;
                this.speechRecognizer.interimResults = true;
                this.speechRecognizer.lang = 'en-US';

                this.speechRecognizer.onresult = (event) => {
                    let fullText = '';
                    for (let i = 0; i < event.results.length; ++i) {
                        fullText += event.results[i][0].transcript + ' ';
                    }
                    this.transcript = fullText.trim();
                };
            } catch (e) {
                console.warn("SpeechRecognition init failed:", e);
            }
        }
    }

    /**
     * Starts recording audio frames and rendering live visualizer on canvas.
     */
    async startRecording(canvasElement = null, vuMeterElement = null) {
        if (this.isRecording) {
            console.warn("ApexVoiceRecorder: startRecording called while already recording.");
            return;
        }

        await this.init();

        this.pcmData = [];
        this.transcript = '';
        this.isRecording = true;

        if (this.speechRecognizer) {
            try {
                this.speechRecognizer.start();
            } catch (e) {}
        }

        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.analyserNode = this.audioContext.createAnalyser();
        this.analyserNode.fftSize = 512;
        this.analyserNode.smoothingTimeConstant = 0.8;

        const bufferSize = 4096;
        this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

        this.processorNode.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const inputBuffer = e.inputBuffer.getChannelData(0);
            this.pcmData.push(new Float32Array(inputBuffer));
        };

        this.sourceNode.connect(this.analyserNode);
        this.analyserNode.connect(this.processorNode);
        this.processorNode.connect(this.audioContext.destination);

        if (canvasElement || vuMeterElement) {
            this.renderVisualizer(canvasElement, vuMeterElement);
        }
    }

    /**
     * Stops recording, downsamples PCM to 16kHz, and encodes into WAV base64.
     */
    async stopRecording() {
        if (!this.isRecording) return null;
        this.isRecording = false;

        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }

        if (this.speechRecognizer) {
            try {
                this.speechRecognizer.stop();
            } catch (e) {}
        }

        if (this.processorNode) {
            try {
                this.processorNode.disconnect();
                this.analyserNode.disconnect();
                this.sourceNode.disconnect();
            } catch (e) {}
        }

        // Release stream tracks so hardware microphone light turns off
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Merge raw float32 chunks
        let totalSamples = 0;
        for (let i = 0; i < this.pcmData.length; i++) {
            totalSamples += this.pcmData[i].length;
        }

        if (totalSamples === 0) {
            return null;
        }

        const mergedFloat32 = new Float32Array(totalSamples);
        let offset = 0;
        for (let i = 0; i < this.pcmData.length; i++) {
            mergedFloat32.set(this.pcmData[i], offset);
            offset += this.pcmData[i].length;
        }

        const nativeSampleRate = this.audioContext ? this.audioContext.sampleRate : 16000;
        const resampledData = this.resampleTo16kHz(mergedFloat32, nativeSampleRate);
        const duration = resampledData.length / this.targetSampleRate;

        const wavBlob = this.encodeWAV(resampledData, this.targetSampleRate);
        const base64Wav = await this.blobToBase64(wavBlob);

        return {
            wavBlob: wavBlob,
            base64Wav: base64Wav,
            transcript: this.transcript,
            durationSec: Math.round(duration * 100) / 100
        };
    }

    /**
     * Linear interpolation resampler to convert arbitrary audioContext sampleRate down to 16,000 Hz.
     */
    resampleTo16kHz(float32Array, inputSampleRate) {
        if (inputSampleRate === this.targetSampleRate || !inputSampleRate) {
            return float32Array;
        }

        const ratio = inputSampleRate / this.targetSampleRate;
        const newLength = Math.round(float32Array.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const originIndex = i * ratio;
            const index1 = Math.floor(originIndex);
            const index2 = Math.min(index1 + 1, float32Array.length - 1);
            const weight = originIndex - index1;
            result[i] = float32Array[index1] * (1 - weight) + float32Array[index2] * weight;
        }

        return result;
    }

    /**
     * Encodes Float32 PCM array into 16-bit Mono WAV Blob.
     */
    encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        this.writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        this.writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        this.writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw PCM) */
        view.setUint16(20, 1, true);
        /* channel count (mono) */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate */
        view.setUint32(28, sampleRate * 2, true);
        /* block align */
        view.setUint16(32, 2, true);
        /* bits per sample */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        this.writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Write 16-bit PCM samples
        let index = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            index += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    /**
     * Renders real-time oscilloscope waveform & frequency spectrum.
     */
    renderVisualizer(canvas, vuMeter) {
        if (!this.analyserNode) return;

        const bufferLength = this.analyserNode.frequencyBinCount;
        const timeData = new Uint8Array(bufferLength);
        const freqData = new Uint8Array(bufferLength);

        const ctx = canvas ? canvas.getContext('2d') : null;

        const draw = () => {
            if (!this.isRecording) return;
            this.animFrameId = requestAnimationFrame(draw);

            this.analyserNode.getByteTimeDomainData(timeData);
            this.analyserNode.getByteFrequencyData(freqData);

            // Compute volume level for VU meter
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const val = (timeData[i] - 128) / 128;
                sum += val * val;
            }
            const rms = Math.sqrt(sum / bufferLength);
            const vuPct = Math.min(100, Math.round(rms * 280));
            if (vuMeter) {
                vuMeter.style.width = `${vuPct}%`;
            }

            // Draw Canvas Visualizer
            if (ctx && canvas) {
                // Handle dynamic screen width / high DPI resizing
                const rect = canvas.getBoundingClientRect();
                if (rect.width > 0 && canvas.width !== Math.floor(rect.width)) {
                    canvas.width = Math.floor(rect.width);
                }
                const width = canvas.width;
                const height = canvas.height;

                ctx.fillStyle = 'rgba(7, 11, 20, 0.5)';
                ctx.fillRect(0, 0, width, height);

                // Draw Frequency Spectrum Bars
                const numBars = 32;
                const barWidth = width / numBars;
                for (let i = 0; i < numBars; i++) {
                    const barHeight = (freqData[i * 2] / 255) * (height * 0.7);
                    const grad = ctx.createLinearGradient(0, height, 0, height - barHeight);
                    grad.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
                    grad.addColorStop(1, 'rgba(6, 182, 212, 0.45)');
                    ctx.fillStyle = grad;
                    ctx.fillRect(i * barWidth, height - barHeight, Math.max(1, barWidth - 2), barHeight);
                }

                // Draw Oscilloscope Glowing Waveform
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = '#10b981';
                ctx.shadowBlur = 8;
                ctx.shadowColor = '#10b981';

                ctx.beginPath();
                const sliceWidth = width / bufferLength;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const v = timeData[i] / 128.0;
                    const y = (v * height) / 2;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    x += sliceWidth;
                }

                ctx.lineTo(width, height / 2);
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
        };

        draw();
    }
}

window.ApexVoiceRecorder = ApexVoiceRecorder;
