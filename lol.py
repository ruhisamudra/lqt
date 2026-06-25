# use this to make the corrected new files

import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.fft import fft, fftfreq

def batch_save_new_corrected_pipeline(recnum, output_prefix="trace_corrected_triangle"):
    # ─── 1. Setup Input and Output Directories ─────────────────────────
    input_folder = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
    # Updated output folder path to 'corr2' as requested
    output_folder = f"data/2025-10-22/{recnum}/timeseries corrected new/" 
    
    os.makedirs(output_folder, exist_ok=True)
    
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    print(f"Found {len(files)} files to process in {input_folder}")

    sampling_rate = 250.0  # 250 fps sampling rate

    for idx, filename in enumerate(files):
        # Load raw trace
        raw_trace = np.load(os.path.join(input_folder, filename))
        n_points = len(raw_trace)
        
        samples_per_ms = n_points / 1000.0
        MIN_APD_MS = 140.0  
        min_apd_frames = int(MIN_APD_MS * samples_per_ms)

        # ─── 2. Normalize ─────────────────────────────────────────────
        raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

        # ─── 3. Automated FFT & Cutoff Frequency Selection ────────────
        fft_vals = np.abs(fft(raw_norm))
        fft_freqs = fftfreq(n_points, d=1.0 / sampling_rate)

        half_points = n_points // 2
        freqs_side = fft_freqs[:half_points]
        vals_side = fft_vals[:half_points]

        fft_slope = np.gradient(vals_side)
        search_start_idx = np.searchsorted(freqs_side, 3.0)
        search_end_idx = np.searchsorted(freqs_side, 50.0)

        optimal_idx = search_start_idx + np.argmax(fft_slope[search_start_idx:search_end_idx])
        detected_cutoff = freqs_side[optimal_idx]

        CUTOFF_FREQUENCY = np.clip(detected_cutoff, 15.0, 45.0)

        # ─── 4. Butterworth Filter Implementation ─────────────────────
        nyquist = 0.5 * sampling_rate
        normal_cutoff = CUTOFF_FREQUENCY / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        raw_smooth_butter = filtfilt(b, a, raw_norm)

        first_deriv_butter = np.gradient(raw_smooth_butter)
        second_deriv_butter = np.gradient(first_deriv_butter)

        min_distance_samples = int(100 * samples_per_ms)
        upstroke_indices, _ = find_peaks(
            first_deriv_butter,
            height=0.05,
            distance=min_distance_samples,
            prominence=0.05
        )

        # ─── 5. Revised Markers Logic (Strict Phase Window Cap) ───────
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

            steep = window_start + np.argmin(first_deriv_butter[window_start:window_end])
            steep_indices.append(steep)

            MAX_REPOL_LOOKAHEAD = 25  
            search_end = min(steep + MAX_REPOL_LOOKAHEAD, window_end)
            
            if search_end > steep:
                repol = steep + np.argmax(second_deriv_butter[steep:search_end])
            else:
                repol = steep
                
            repol_indices.append(repol)

        # ─── 6. Savitzky-Golay Pipeline (Baseline Subtraction) ────────
        baseline_trace = raw_norm.copy()
        for up, re in zip(upstroke_indices, repol_indices):
            baseline_trace[up:re] = np.nan

        nan_mask = np.isnan(baseline_trace)
        x_all = np.arange(n_points)
        
        # Fallback if no peaks were detected to avoid crash
        if not np.any(nan_mask):
            corrected_trace_savgol = raw_norm - savgol_filter(raw_norm, window_length=21, polyorder=3)
        else:
            x_valid = x_all[~nan_mask]
            y_valid = baseline_trace[~nan_mask]
            baseline_interpolated = np.interp(x_all, x_valid, y_valid)
            baseline_smooth_savgol = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
            
            # Final clean data trace from the new pipeline
            corrected_trace_savgol = raw_norm - baseline_smooth_savgol

        # ─── 7. Save Array Outputs as corrected float64 ────────────────
        output_filename = f"{output_prefix}_{idx:04d}.npy"
        np.save(os.path.join(output_folder, output_filename), corrected_trace_savgol)

    print(f"🎉 Complete! Saved {len(files)} new-pipeline corrected files to: {output_folder}")

# Execute the batch pipeline for your recording
batch_save_new_corrected_pipeline("rec08")