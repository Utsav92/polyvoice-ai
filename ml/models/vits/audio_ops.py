"""STFT / mel utilities usable on CPU, CUDA and MPS."""
import torch, librosa

class Spectral(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=1024, hop=256, win=1024, n_mels=80, fmin=0.0, fmax=None):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax or sr / 2)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float(), persistent=False)
    def linear(self, y):  # y [b,1,t] -> magnitude [b,n_fft/2+1,frames]; always fp32 (STFT is unstable in half precision)
        with torch.autocast(device_type=y.device.type, enabled=False): return self._linear(y.float())
    def _linear(self, y):
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(y, (pad, pad), mode="reflect").squeeze(1)
        s = torch.stft(y, self.n_fft, self.hop, self.win, self.window, center=False, return_complex=True)
        return torch.sqrt(s.real ** 2 + s.imag ** 2 + 1e-6)
    def mel(self, y):
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, self.linear(y)), min=1e-5))
    def mel_from_linear(self, spec):
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, spec), min=1e-5))
