# TIME SERIES PLOTS FOR APD DETECTION

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.fft import fft, fftfreq

def run_apd_plots(version, recnum, filenum):

    # Plots from OLD pipeline

    if version == "old":
        # ─── Settings ─────────────────────────────────────────────────────────── 
        # Load all traces
        folder = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
        files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        all_traces = np.array([np.load(os.path.join(folder, f)) for f in files])

        # Raw individual trace normalized
        raw_trace = all_traces[filenum]
        raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

        # Time axis in ms
        n_points = len(raw_norm)
        time_ms = np.linspace(0, 1000, n_points)

        # Smooth the trace for detection of upstroke and repolarization
        raw_smooth = savgol_filter(raw_norm, window_length = 11, polyorder = 3)

        # First and second derivatives
        first_deriv = savgol_filter(np.gradient(raw_smooth), window_length = 11, polyorder = 3)
        second_deriv = savgol_filter(np.gradient(first_deriv), window_length = 11, polyorder = 3)

        # Automatically detect all upstrokes = peaks in first derivative
        samples_per_ms = n_points / 1000
        min_distance_samples = int(100 * samples_per_ms)

        upstroke_indices, _ = find_peaks(
            first_deriv, 
            height=0.05,
            distance=min_distance_samples,
            prominence=0.05
        )

        # ─── PANELS 1 & 2: Upstroke, Downstroke, Repolarization ───────────────────────────────────────────────────────────

        steep_indices = []
        repol_indices = []

        for i, up_idx in enumerate(upstroke_indices):
            window_start = up_idx + 5
            
            if i + 1 < len(upstroke_indices):
                window_end = upstroke_indices[i + 1]
            else:
                window_end = len(raw_smooth)
            
            # Steep downslope = local minimum of the first derivative of action potential
            steep = window_start + np.argmin(first_deriv[window_start:window_end])
            steep_indices.append(steep)
            
            # Repolarization = local maximum of the second derivative of action potential
            search_end = min(steep + 30, window_end)  # only look 30 samples after steep
            repol = steep + np.argmax(np.abs(second_deriv[steep:search_end]))
            repol_indices.append(repol)

        # ─── PANEL 3: Corrected Baseline on Raw Trace ───────────────────────────────────────────────────────────

        # Set Nan between each upstroke and repolarization
        baseline_trace = raw_norm.copy()
        for up, re in zip(upstroke_indices, repol_indices):
            baseline_trace[up:re] = np.nan

        # Linearly interpolate across NaN (upstroke to repolarization becomes straight line)
        nan_mask = np.isnan(baseline_trace)
        x_all = np.arange(n_points)
        x_valid = x_all[~nan_mask]
        y_valid = baseline_trace[~nan_mask]
        baseline_interpolated = np.interp(x_all, x_valid, y_valid)

        # Smooth interpolated baseline using savgol
        baseline_smooth = savgol_filter(baseline_interpolated, window_length = 21, polyorder = 3)

        # ─── PANEL 4: Corrected Trace after Baseline Subtraction ───────────────────────────────────────────────────────────

        # Subtract the fitted baseline from the original trace
        corrected_trace = raw_norm - baseline_smooth

        # ─── Plot all Panels ───────────────────────────────────────────────────────────

        fig, axes = plt.subplots(4, 1, figsize = (10,10), sharex = False)

        # Panel 1: Raw Signal
        axes[0].plot(time_ms, raw_norm, color = "black", linewidth = 1.5)

        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[0].axvline(time_ms[up], color = "red", linewidth = 1.5)
            axes[0].axvline(time_ms[st], color = "magenta", linewidth = 1.5)
            axes[0].axvline(time_ms[re], color = "blue", linewidth = 1.5)

        axes[0].set_xlabel("Time [ms]")
        axes[0].set_ylabel("Normalized Signal")

        # Panel 2: Derivatives
        axes[1].plot(time_ms, first_deriv, color = "black", linewidth = 1.5,)
        axes[1].plot(time_ms, second_deriv, color = "black", linewidth = 1.5)
        axes[1].axhline(0, color = "gray", linewidth=1)

        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[1].axvline(time_ms[up], color = "red", linewidth = 1.5)
            axes[1].axvline(time_ms[st], color = "magenta", linewidth = 1.5)
            axes[1].axvline(time_ms[re], color = "blue", linewidth = 1.5)

        axes[1].set_xlabel("Time [ms]")
        axes[1].set_ylabel("dF/dt")

        # Panel 3: Corrected Baseline on Raw Trace
        axes[2].plot(time_ms, raw_norm, color = "black", linewidth = 1.5, label = "Raw Trace")
        axes[2].plot(time_ms, baseline_smooth, color = "green", linewidth = 1.5, label = "Fitted Baseline")

        axes[2].set_xlabel("Time [ms]")
        axes[2].set_ylabel("Normalized Signal")
        axes[2].legend(loc = "upper right", fontsize = 8)

        # Panel 4: Corrected Trace after Baseline Subtraction
        axes[3].plot(time_ms, corrected_trace, color = "black", linewidth = 1.5)

        axes[3].set_xlabel("Time [ms]")
        axes[3].set_ylabel("Corrected Signal")

        # Final Layout
        plt.tight_layout()
        plt.savefig("four_panel_plot.png", dpi = 300, bbox_inches = "tight")
        plt.show()

        APD = time_ms[repol_indices] - time_ms[upstroke_indices]
        print("APD per beat (ms):", APD)

    # Plots from NEW pipeline

    elif version == "new":
        # ─── Settings ───────────────────────────────────────────────────────────

        folder = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
        files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

        raw_trace = np.load(os.path.join(folder, files[filenum]))

        n_points = len(raw_trace)
        time_ms = np.linspace(0, 1000, n_points)
        samples_per_ms = n_points / 1000.0
        sampling_rate = 250.0  # 250 fps sampling rate

        # PHYSIOLOGICAL SAFETY CONSTRAINT
        MIN_APD_MS = 140.0  
        min_apd_frames = int(MIN_APD_MS * samples_per_ms)

        # ─── Normalize ───────────────────────────────────────────────────────────

        raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

        # ─── Automated FFT & Cutoff Frequency Selection (Panels 1 & 2) ───────────────────────────────────────────────────────────

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
        print(f"--> Automated FFT Analysis complete. Optimal Cutoff Selected: {CUTOFF_FREQUENCY:.2f} Hz")


        # ─── Butterworth Filter Implementation (Panels 1 & 2) ───────────────────────────────────────────────────────────

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)

        raw_smooth_butter = butter_lowpass_filter(raw_norm, cutoff=CUTOFF_FREQUENCY, fs=sampling_rate, order=4)

        first_deriv_butter = np.gradient(raw_smooth_butter)
        second_deriv_butter = np.gradient(first_deriv_butter)

        min_distance_samples = int(100 * samples_per_ms)
        upstroke_indices, _ = find_peaks(
            first_deriv_butter,
            height = 0.05,
            distance = min_distance_samples,
            prominence = 0.05
        )

        # ─── REVISED Markers Logic (Strict Phase Window Cap) ───────────────────────────────────────────────────────────

        steep_indices = []
        repol_indices = []

        # Define the maximum time (in ms) allowed for an action potential phase before the dip
        # Looking at your trace, 350-400 ms perfectly covers the true signal body.
        MAX_PHASE_WINDOW_MS = 500.0  
        max_phase_frames = int(MAX_PHASE_WINDOW_MS * samples_per_ms)

        for i, up_idx in enumerate(upstroke_indices):
            window_start = up_idx + min_apd_frames
            
            # Identify the absolute limit (the next upstroke)
            if i + 1 < len(upstroke_indices):
                next_upstroke = upstroke_indices[i + 1]
            else:
                next_upstroke = n_points - 1

            # Cap the search window tightly so it cannot look past the main AP body
            window_end = min(up_idx + max_phase_frames, next_upstroke)
            window_end = min(window_end, n_points - 1)
            
            if window_start >= window_end:
                window_start = up_idx + 5

            # 1. Steep downslope = local minimum of the first derivative within our capped window
            steep = window_start + np.argmin(first_deriv_butter[window_start:window_end])
            steep_indices.append(steep)

            # 2. Repolarization = local maximum of second derivative within the safe window
            MAX_REPOL_LOOKAHEAD = 25  # Allow a slightly wider search immediately after our steep point
            search_end = min(steep + MAX_REPOL_LOOKAHEAD, window_end)
            
            if search_end > steep:
                # Using raw second derivative to capture the local inflection change at the plateau base
                repol = steep + np.argmax(second_deriv_butter[steep:search_end])
            else:
                repol = steep
                
            repol_indices.append(repol)

        # ─── Original Savitzky-Golay Pipeline (Panels 3 & 4) ───────────────────────────────────────────────────────────

        baseline_trace = raw_norm.copy()
        for up, re in zip(upstroke_indices, repol_indices):
            baseline_trace[up:re] = np.nan

        nan_mask = np.isnan(baseline_trace)
        x_all = np.arange(n_points)
        x_valid = x_all[~nan_mask]
        y_valid = baseline_trace[~nan_mask]
        baseline_interpolated = np.interp(x_all, x_valid, y_valid)

        baseline_smooth_savgol = savgol_filter(baseline_interpolated, window_length = 21, polyorder = 3)
        corrected_trace_savgol = raw_norm - baseline_smooth_savgol


        # ─── Calculate Metrics ───────────────────────────────────────────────────────────

        APD = time_ms[repol_indices] - time_ms[upstroke_indices]
        print("Bounded APD per beat (ms):", APD)


        # ─── Plot all Panels ───────────────────────────────────────────────────────────

        fig, axes = plt.subplots(4, 1, figsize = (10, 10), sharex = False)

        # Panel 1: Raw Signal with Markers
        axes[0].plot(time_ms, raw_norm, color = "lightgray", linewidth = 1, label = "Raw Noise")
        axes[0].plot(time_ms, raw_smooth_butter, color = "black", linewidth = 1.5, label = "Butterworth Filtered")
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[0].axvline(time_ms[up], color = "red", linewidth = 1.5)
            axes[0].axvline(time_ms[st], color = "magenta", linewidth = 1.5)
            axes[0].axvline(time_ms[re], color = "blue", linewidth = 1.5)
        axes[0].set_xlabel("Time [ms]")
        axes[0].set_ylabel("Normalized Signal")
        axes[0].legend(loc="upper right")

        # Panel 2: Butterworth Derivatives
        axes[1].plot(time_ms, first_deriv_butter, color = "black", linewidth = 1.5)
        axes[1].plot(time_ms, second_deriv_butter, color = "black", linewidth = 1.5)
        axes[1].axhline(0, color = "gray", linewidth = 1)
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[1].axvline(time_ms[up], color = "red", linewidth = 1.5)
            axes[1].axvline(time_ms[st], color = "magenta", linewidth = 1.5)
            axes[1].axvline(time_ms[re], color = "blue", linewidth = 1.5)
        axes[1].set_xlabel("Time [ms]")
        axes[1].set_ylabel("dF/dt")

        # Panel 3: Corrected Baseline on Raw Trace
        axes[2].plot(time_ms, raw_norm, color = "black", linewidth = 1.5, label = "Raw Trace")
        axes[2].plot(time_ms, baseline_smooth_savgol, color = "green", linewidth = 1.5, label = "Fitted Baseline (SavGol)")
        axes[2].set_xlabel("Time [ms]")
        axes[2].set_ylabel("Normalized Signal")
        axes[2].legend(loc = "upper right", fontsize = 8)

        # Panel 4: Corrected Trace Only (Clean)
        axes[3].plot(time_ms, corrected_trace_savgol, color = "black", linewidth = 1.5)
        axes[3].set_xlabel("Time [ms]")
        axes[3].set_ylabel("Corrected Signal")

        plt.tight_layout()
        plt.savefig("four_panel_plot.png", dpi = 300, bbox_inches = "tight")
        plt.show()

        # ─── Separate Standalone Fourier Transform Diagnostic Plot ───────────────────────────────────────────────────────────

        plt.figure(figsize=(8, 4))
        plt.plot(freqs_side, vals_side, color="black", linewidth=1.5, label="FFT Power Spectrum")
        plt.axvline(CUTOFF_FREQUENCY, color="red", linestyle="--", linewidth=1.5, label=f"Auto-Cutoff ({CUTOFF_FREQUENCY:.2f} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum & Automated Filter Selection")
        plt.xlim(0, 125)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


# Call the old or new pipeline, recording, and trace
run_apd_plots("old", "rec03", 46)