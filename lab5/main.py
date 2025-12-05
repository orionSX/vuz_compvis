import numpy as np
import scipy.io.wavfile as wav


class AudioDenoiser:
    def __init__(self, window_size=1024, hop_size=256):
        self.window_size = window_size
        self.hop_size = hop_size

    def _fft(self, signal):
        signal = np.asarray(signal, dtype=complex)
        length = signal.shape[0]

        if length <= 1:
            return signal

        even = self._fft(signal[::2])
        odd = self._fft(signal[1::2])
        factors = np.exp(-2j * np.pi * np.arange(length) / length)

        return np.concatenate(
            [even + factors[: length // 2] * odd, even - factors[: length // 2] * odd]
        )

    def _ifft(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        return np.conjugate(self._fft(np.conjugate(spectrum))) / len(spectrum)

    def _to_spectrogram(self, signal):
        window = np.hanning(self.window_size)
        frames = []

        for start in range(0, len(signal) - self.window_size, self.hop_size):
            frame = signal[start : start + self.window_size] * window
            frames.append(self._fft(frame))

        return np.array(frames), window

    def _from_spectrogram(self, spectrogram, window):
        output_length = self.hop_size * (len(spectrogram) + 1) + self.window_size
        result = np.zeros(output_length)
        window_sum = np.zeros(output_length)

        idx = 0
        for frame_fft in spectrogram:
            frame = np.real(self._ifft(frame_fft))

            result[idx : idx + self.window_size] += frame * window
            window_sum[idx : idx + self.window_size] += window**2
            idx += self.hop_size

        nonzero = window_sum > 1e-8
        result[nonzero] /= window_sum[nonzero]

        return result

    def process(
        self,
        input_file,
        output_file,
        noise_dur_sec,
        noise_reduction_strength,
        min_signal_level,
    ):
        sample_rate, audio_data = wav.read(input_file)
        noise_frames = int(noise_dur_sec * sample_rate / self.hop_size)
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        spectrogram, window = self._to_spectrogram(audio_data)
        magnitudes = np.abs(spectrogram)
        phases = np.angle(spectrogram)

        noise_profile = np.mean(magnitudes[:noise_frames, :], axis=0)

        epsilon = 1e-8
        clean_magnitudes = (
            magnitudes - noise_reduction_strength * noise_profile[np.newaxis, :]
        )

        gain = np.maximum(clean_magnitudes / (magnitudes + epsilon), min_signal_level)

        clean_spectrum = gain * magnitudes * np.exp(1j * phases)

        cleaned_audio = self._from_spectrogram(clean_spectrum, window)

        cleaned_audio = np.clip(cleaned_audio, -1, 1)
        cleaned_audio = (cleaned_audio * 32767).astype(np.int16)

        wav.write(output_file, sample_rate, cleaned_audio)
        print(f"Saved: {output_file}")


denoiser = AudioDenoiser()
denoiser.process(
    "D:\\Github\\compvis\\lab5\\v1.wav",
    "D:\\Github\\compvis\\lab5\\v2.wav",
    noise_dur_sec=9,
    noise_reduction_strength=10,
    min_signal_level=0.01,
)
