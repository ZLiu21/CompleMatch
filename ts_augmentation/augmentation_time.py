import random
import numpy as np
import torch
from scipy.interpolate import CubicSpline, interp1d


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling_1d(x, sigma=0.1):
    factor = np.random.normal(loc=1., scale=sigma, size=x.shape[0])
    return np.multiply(x, factor)


def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))
    x_ = np.multiply(x, factor[:, :])
    return x_


def negated(x):
    return x * -1


def time_flipped_1d(x):
    inv_idx = torch.arange(x.shape[0] - 1, -1, -1).long()
    return x[inv_idx]


def time_flipped_batch(x):
    end_x = []
    for _x in x:
        _xi = time_flipped_1d(_x)
        end_x.append(_xi)

    return np.array(end_x)


def permutation_1d(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[0])

    if max_segments > x.shape[0]:
        max_segments = x.shape[0]

    num_segs = np.random.randint(1, max_segments + 1)

    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[0] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)

        warp = np.concatenate(np.random.permutation(splits)).ravel()
        return x[warp]
    else:
        return x


def magnitude_warp_1d(x, sigma=0.2, knot=4):
    orig_steps = np.arange(len(x))

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
    warp_steps = np.linspace(0, len(x) - 1., num=knot + 2)

    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    return x * warper


def magnitude_warp_batch(x, sigma=0.2, knot=4):
    end_x = []
    for _x in x:
        _xi = magnitude_warp_1d(_x, sigma=sigma)
        end_x.append(_xi)

    return np.array(end_x)


def time_warp_1d(x, sigma=0.2, knot=4):
    orig_steps = np.arange(len(x))

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
    warp_steps = np.linspace(0, len(x) - 1., num=knot + 2)

    time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
    scale = (len(x) - 1) / time_warp[-1]
    return np.interp(orig_steps, np.clip(scale * time_warp, 0, len(x) - 1), x)


def time_warp_batch(x, sigma=0.2, knot=4):
    end_x = []
    for _x in x:
        _xi = time_warp_1d(_x, sigma=sigma)
        end_x.append(_xi)

    return np.array(end_x)


def window_slice_1d(x, reduce_ratio=0.9):
    # Calculate the target length of the sliced window
    target_len = int(np.ceil(reduce_ratio * len(x)))

    # If the target length is not smaller than the original length, return the original array
    if target_len >= len(x):
        return x

    # Randomly choose the start point for slicing
    start = np.random.randint(0, len(x) - target_len)

    # Calculate the end point for slicing
    end = start + target_len

    # Interpolate the sliced window back to the original length
    return np.interp(np.linspace(0, target_len, num=len(x)), np.arange(target_len), x[start:end])


def window_slice_batch(x, reduce_ratio=0.9):
    end_x = []
    for _x in x:
        _xi = window_slice_1d(_x, reduce_ratio=reduce_ratio)
        end_x.append(_xi)

    return np.array(end_x)


def resample_1d(x):
    orig_steps = np.arange(len(x)) 
    interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / 3)  

    Interp = interp1d(orig_steps, x, kind='linear')
    InterpVal = Interp(interp_steps)  

    start = random.choice(orig_steps)
    resample_index = np.arange(start, 3 * len(x), 2)[:len(x)] 

    return InterpVal[resample_index]  


def resample_batch(x):
    end_x = []
    for _x in x:
        _xi = resample_1d(_x)
        end_x.append(_xi)

    return np.array(end_x)


### TS2Vec mask aug1
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def crop_ts_left_right(sample, temporal_unit=0):
    ts_l = sample.size(1)
    crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
    crop_left = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l

    crop_eleft = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=sample.size(0))

    out1 = take_per_row(sample, crop_offset + crop_eleft, crop_right - crop_eleft)
    out1 = out1[:, -crop_l:]

    out2 = take_per_row(sample, crop_offset + crop_left, crop_eright - crop_left)
    out2 = out2[:, :crop_l]

    return out1, out2
