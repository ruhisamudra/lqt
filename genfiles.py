# SAVES TIMESERIES DETRENDED .npy FILES FROM RATIOMETRIC RAW

import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter

# ─── Settings ──────────────────────────────────────────────────────────────

recnum = "rec08" # change here
input_folder  = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
output_folder = f"data/2025-10-22/{recnum}/timeseries detrended/"

os.makedirs(output_folder, exist_ok = True)

files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
print(f"Found {len(files)} traces to process...")

# ─── Core Function (Old Pipeline) ─────────────────────────────────────────

def detrend_trace(raw_trace):

    n_points = len(raw_trace)
    samples_per_ms = n_points / 1000.0

    # Normalize
    raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

    # Smooth for detection
    raw_smooth = savgol_filter(raw_norm, window_length = 11, polyorder = 3)

    # First and second derivatives
    first_deriv  = savgol_filter(np.gradient(raw_smooth),  window_length = 11, polyorder = 3)
    second_deriv = savgol_filter(np.gradient(first_deriv), window_length = 11, polyorder = 3)

    # Detect upstrokes
    min_distance_samples = int(100 * samples_per_ms)
    upstroke_indices, _ = find_peaks(
        first_deriv,
        height     = 0.05,
        distance   = min_distance_samples,
        prominence = 0.05
    )

    # Detect steep downslope and repolarization for each beat
    steep_indices = []
    repol_indices = []

    for i, up_idx in enumerate(upstroke_indices):
        window_start = up_idx + 5

        if i + 1 < len(upstroke_indices):
            window_end = upstroke_indices[i + 1]
        else:
            window_end = len(raw_smooth)

        # Safety check: skip this beat if window is empty or invalid
        if window_start >= window_end:
            continue

        steep = window_start + np.argmin(first_deriv[window_start:window_end])
        steep_indices.append(steep)

        search_end = min(steep + 30, window_end)

        # Safety check again before second argmax
        if search_end <= steep:
            repol_indices.append(steep)
            continue

        repol = steep + np.argmax(np.abs(second_deriv[steep:search_end]))
        repol_indices.append(repol)

    # Baseline correction
    baseline_trace = raw_norm.copy()
    for up, re in zip(upstroke_indices, repol_indices):
        baseline_trace[up:re] = np.nan

    nan_mask = np.isnan(baseline_trace)
    x_all   = np.arange(n_points)
    x_valid = x_all[~nan_mask]
    y_valid = baseline_trace[~nan_mask]

    if len(x_valid) < 2:
        return raw_norm  # fallback if correction not possible

    baseline_interpolated = np.interp(x_all, x_valid, y_valid)
    baseline_smooth = savgol_filter(baseline_interpolated, window_length = 21, polyorder = 3)
    corrected_trace = raw_norm - baseline_smooth

    return corrected_trace

# ─── Batch Processing ──────────────────────────────────────────────────────

for idx, filename in enumerate(files):

    raw_trace = np.load(os.path.join(input_folder, filename))
    detrended_trace = detrend_trace(raw_trace)

    output_filename = f"trace_detrended_triangle_{idx:04d}.npy"
    np.save(os.path.join(output_folder, output_filename), detrended_trace)

    if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
        print(f"Progress: {idx + 1} / {len(files)}")

print("Done! Detrended traces saved to:", output_folder)



# SAVE TIMESERIES CORRECTED NEW .npy FILES FROM RATIOMETRIC RAW

import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.fft import fft, fftfreq

# ─── Settings ──────────────────────────────────────────────────────────────

recnum = "rec08" # change here
input_folder  = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
output_folder = f"data/2025-10-22/{recnum}/timeseries corrected new/"

os.makedirs(output_folder, exist_ok = True)

files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
print(f"Found {len(files)} traces to process...")

# ─── Helper Function ──────────────────────────────────────────────────────

def butter_lowpass_filter(data, cutoff, fs, order = 4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    return filtfilt(b, a, data)

# ─── Core Function (New Pipeline) ─────────────────────────────────────────

def correct_trace_new(raw_trace):

    n_points = len(raw_trace)
    samples_per_ms = n_points / 1000.0
    sampling_rate = 250.0

    MIN_APD_MS = 140.0
    min_apd_frames = int(MIN_APD_MS * samples_per_ms)

    # Normalize
    raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

    # Automated FFT cutoff selection
    fft_vals  = np.abs(fft(raw_norm))
    fft_freqs = fftfreq(n_points, d = 1.0 / sampling_rate)
    half_points = n_points // 2
    freqs_side = fft_freqs[:half_points]
    vals_side  = fft_vals[:half_points]

    fft_slope = np.gradient(vals_side)
    search_start_idx = np.searchsorted(freqs_side, 3.0)
    search_end_idx   = np.searchsorted(freqs_side, 50.0)

    if search_end_idx <= search_start_idx:
        CUTOFF_FREQUENCY = 30.0  # fallback
    else:
        optimal_idx = search_start_idx + np.argmax(fft_slope[search_start_idx:search_end_idx])
        detected_cutoff = freqs_side[optimal_idx]
        CUTOFF_FREQUENCY = np.clip(detected_cutoff, 15.0, 45.0)

    # Butterworth filter and derivatives
    raw_smooth_butter  = butter_lowpass_filter(raw_norm, cutoff = CUTOFF_FREQUENCY, fs = sampling_rate, order = 4)
    first_deriv_butter  = np.gradient(raw_smooth_butter)
    second_deriv_butter = np.gradient(first_deriv_butter)

    # Detect upstrokes
    min_distance_samples = int(100 * samples_per_ms)
    upstroke_indices, _ = find_peaks(
        first_deriv_butter,
        height     = 0.05,
        distance   = min_distance_samples,
        prominence = 0.05
    )

    # Detect steep downslope and repolarization
    steep_indices = []
    repol_indices = []

    MAX_PHASE_WINDOW_MS = 500.0
    max_phase_frames = int(MAX_PHASE_WINDOW_MS * samples_per_ms)

    for i, up_idx in enumerate(upstroke_indices):
        window_start = up_idx + min_apd_frames

        if i + 1 < len(upstroke_indices):
            next_upstroke = upstroke_indices[i + 1]
        else:
            next_upstroke = n_points - 1

        window_end = min(up_idx + max_phase_frames, next_upstroke, n_points - 1)

        if window_start >= window_end:
            window_start = up_idx + 5

        if window_start >= window_end:
            continue

        steep = window_start + np.argmin(first_deriv_butter[window_start:window_end])
        steep_indices.append(steep)

        search_end = min(steep + 25, window_end)

        if search_end <= steep:
            repol_indices.append(steep)
            continue

        repol = steep + np.argmax(second_deriv_butter[steep:search_end])
        repol_indices.append(repol)

    # Baseline correction (Savitzky-Golay, same as old pipeline)
    baseline_trace = raw_norm.copy()
    for up, re in zip(upstroke_indices, repol_indices):
        baseline_trace[up:re] = np.nan

    nan_mask = np.isnan(baseline_trace)
    x_all   = np.arange(n_points)
    x_valid = x_all[~nan_mask]
    y_valid = baseline_trace[~nan_mask]

    if len(x_valid) < 2:
        return raw_norm  # fallback if correction not possible

    baseline_interpolated = np.interp(x_all, x_valid, y_valid)
    baseline_smooth = savgol_filter(baseline_interpolated, window_length = 21, polyorder = 3)
    corrected_trace = raw_norm - baseline_smooth

    return corrected_trace

# ─── Batch Processing ──────────────────────────────────────────────────────

for idx, filename in enumerate(files):

    raw_trace = np.load(os.path.join(input_folder, filename))
    corrected_trace = correct_trace_new(raw_trace)

    output_filename = f"trace_corrected_triangle_{idx:04d}.npy"
    np.save(os.path.join(output_folder, output_filename), corrected_trace)

    if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
        print(f"Progress: {idx + 1} / {len(files)}")

print("Done! Corrected traces saved to:", output_folder)