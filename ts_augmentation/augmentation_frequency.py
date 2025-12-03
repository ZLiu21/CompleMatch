import numpy as np
import torch
from torch.fft import fft, ifft


def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F


def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.FloatTensor(x.shape).uniform_() > pertub_ratio  # maskout_ratio are False
    return x * mask


def add_frequency(x, pertub_ratio=0.0):
    mask = torch.FloatTensor(x.shape).uniform_() > (1 - pertub_ratio)  # only pertub_ratio of all values are True
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am
    return x + pertub_matrix


def data_augmentation_frequency_domain(sample, q_magnitude=0.1, delta_phase=0.1):
    freq_data = fft(sample)
    magnitude = torch.abs(freq_data)
    phase = torch.angle(freq_data)
    mean_magnitude = magnitude.mean().item()
    std_magnitude = magnitude.std().item()
    perturbation_magnitude = torch.normal(mean_magnitude, q_magnitude * std_magnitude, size=magnitude.shape)
    augmented_magnitude = magnitude + perturbation_magnitude

    perturbation_phase = torch.normal(0, delta_phase, size=phase.shape)
    augmented_phase = phase + perturbation_phase

    augmented_freq_data = augmented_magnitude * (torch.cos(augmented_phase) + 1j * torch.sin(augmented_phase))

    augmented_data = ifft(augmented_freq_data).real.numpy()
    return torch.tensor(augmented_data)


def ifft_phase_shift_1d(sample):
    fd = torch.fft.fftshift(torch.fft.fft(sample))

    amp = fd.abs()
    phase = fd.angle()

    angles = np.random.uniform(low=-np.pi, high=np.pi, size=sample.shape)
    phase = phase + torch.tensor(angles, dtype=phase.dtype)

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.real(torch.fft.ifft(torch.fft.ifftshift(cmp)))

    return ifft


def highpass_filter(sample, relative_cutoff): ###  relative_cutoff is in [0, 1]
    n = sample.shape[0]

    fft_result = torch.fft.fft(sample)
    freq = torch.fft.fftfreq(n)

    highpass_mask = torch.abs(freq) > relative_cutoff
    filtered_fft = fft_result * highpass_mask.type(sample.dtype)

    ifft_result = torch.fft.ifft(filtered_fft)

    return torch.real(ifft_result)


def highpass_filter_batch(sample, relative_cutoff=0.1):
    end_x = []
    for _x in sample:
        _xi = highpass_filter(sample=_x, relative_cutoff=relative_cutoff)
        end_x.append(_xi)

    return torch.stack(end_x)


def lowpass_filter(sample, relative_cutoff):
    n = sample.shape[0]

    fft_result = torch.fft.fft(sample)
    freq = torch.fft.fftfreq(n)

    lowpass_mask = torch.abs(freq) < relative_cutoff
    filtered_fft = fft_result * lowpass_mask.type(sample.dtype)

    ifft_result = torch.fft.ifft(filtered_fft)

    return torch.real(ifft_result)


def lowpass_filter_batch(sample, relative_cutoff=0.1):
    end_x = []
    for _x in sample:
        _xi = lowpass_filter(sample=_x, relative_cutoff=relative_cutoff)
        end_x.append(_xi)

    return torch.stack(end_x)


def gen_new_aug(sample, alpha=0.9):  
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    mixing_coeff = (alpha - 1) * torch.rand(1) + 1
    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * mixing_coeff + (1 - mixing_coeff) * abs_fft[index]
    z = torch.polar(mixed_abs, phase_fft)  # Go back to fft
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time
