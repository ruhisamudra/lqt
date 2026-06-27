# STEP 1: GENERATE PLOTS AND CSV OF APD VALUES

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.fft import fft, fftfreq


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def process_single_trace(file_path, filename, output_plot_dir):
    """
    Processes a single .npy trace file, performs signal processing,
    extracts up to two beat APDs separated by specific time windows,
    and saves a 4-panel diagnostic plot.

    Returns: (beat1_apd, beat2_apd) or (-1, -1) if an execution error occurs.
    """
    try:
        raw_trace = np.load(file_path)
        n_points = len(raw_trace)
        time_ms = np.linspace(0, 1000, n_points)
        samples_per_ms = n_points / 1000.0
        sampling_rate = 250.0

        MIN_APD_MS = 140.0
        min_apd_frames = int(MIN_APD_MS * samples_per_ms)
        MAX_PHASE_WINDOW_MS = 600.0
        max_phase_frames = int(MAX_PHASE_WINDOW_MS * samples_per_ms)

        raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

        # Automated FFT cutoff selection
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
        CUTOFF_FREQUENCY = float(np.clip(detected_cutoff, 15.0, 45.0))

        # Filter and derivatives
        raw_smooth_butter = butter_lowpass_filter(raw_norm, cutoff=CUTOFF_FREQUENCY, fs=sampling_rate, order=4)
        first_deriv_butter = np.gradient(raw_smooth_butter)
        second_deriv_butter = np.gradient(first_deriv_butter)

        min_distance_samples = int(100 * samples_per_ms)
        upstroke_indices, _ = find_peaks(
            first_deriv_butter,
            height=0.05,
            distance=min_distance_samples,
            prominence=0.05
        )

        steep_indices = []
        repol_indices = []

        for i, up_idx in enumerate(upstroke_indices):
            window_start = up_idx + min_apd_frames
            if i + 1 < len(upstroke_indices):
                next_upstroke = upstroke_indices[i + 1]
            else:
                next_upstroke = n_points - 1

            window_end = min(up_idx + max_phase_frames, next_upstroke)
            window_end = min(window_end, n_points - 1)

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

        # Baseline correction via Savitzky-Golay
        baseline_trace = raw_norm.copy().astype(float)
        for up, re in zip(upstroke_indices, repol_indices):
            baseline_trace[up:re] = np.nan

        nan_mask = np.isnan(baseline_trace)
        x_all = np.arange(n_points)
        x_valid = x_all[~nan_mask]
        y_valid = baseline_trace[~nan_mask]

        if len(x_valid) < 2:
            return "N/A", "N/A"

        baseline_interpolated = np.interp(x_all, x_valid, y_valid)
        baseline_smooth_savgol = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
        corrected_trace_savgol = raw_norm - baseline_smooth_savgol

        # ── FIXED: APD assignment — use indexed time_ms, not the full array ──
        beat1_apd = "N/A"
        beat2_apd = "N/A"

        for up_idx, repol_idx in zip(upstroke_indices, repol_indices):
            upstroke_time_ms = time_ms[up_idx]        # scalar, not array
            calculated_apd = time_ms[repol_idx] - time_ms[up_idx]   # scalar

            if 0.0 <= upstroke_time_ms < 575.0:
                beat1_apd = round(float(calculated_apd), 2)
            elif 575.0 <= upstroke_time_ms <= 1000.0:
                beat2_apd = round(float(calculated_apd), 2)

        # 4-panel plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)

        axes[0].plot(time_ms, raw_norm, color="lightgray", linewidth=1, label="Raw Noise")
        axes[0].plot(time_ms, raw_smooth_butter, color="black", linewidth=1.5, label="Butterworth Filtered")
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[0].axvline(time_ms[up], color="red", linewidth=1.5)
            axes[0].axvline(time_ms[st], color="magenta", linewidth=1.5)
            axes[0].axvline(time_ms[re], color="blue", linewidth=1.5)
        axes[0].set_xlabel("Time [ms]")
        axes[0].set_ylabel("Normalized Signal")
        axes[0].legend(loc="upper right")

        axes[1].plot(time_ms, first_deriv_butter, color="black", linewidth=1.5, label="dF/dt")
        axes[1].plot(time_ms, second_deriv_butter, color="gray", linewidth=1.0, label="d²F/dt²")
        axes[1].axhline(0, color="gray", linewidth=1)
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[1].axvline(time_ms[up], color="red", linewidth=1.5)
            axes[1].axvline(time_ms[st], color="magenta", linewidth=1.5)
            axes[1].axvline(time_ms[re], color="blue", linewidth=1.5)
        axes[1].set_xlabel("Time [ms]")
        axes[1].set_ylabel("dF/dt")
        axes[1].legend(loc="upper right")

        axes[2].plot(time_ms, raw_norm, color="black", linewidth=1.5, label="Raw Trace")
        axes[2].plot(time_ms, baseline_smooth_savgol, color="green", linewidth=1.5, label="Fitted Baseline (SavGol)")
        axes[2].set_xlabel("Time [ms]")
        axes[2].set_ylabel("Normalized Signal")
        axes[2].legend(loc="upper right", fontsize=8)

        axes[3].plot(time_ms, corrected_trace_savgol, color="black", linewidth=1.5)
        axes[3].set_xlabel("Time [ms]")
        axes[3].set_ylabel("Corrected Signal")

        # Annotate APD values on panel 0
        apd_label = f"Beat1 APD: {beat1_apd} ms  |  Beat2 APD: {beat2_apd} ms"
        axes[0].set_title(apd_label, fontsize=10)

        plt.tight_layout()

        plot_filename = filename.replace(".npy", "_4panel.png")
        plt.savefig(os.path.join(output_plot_dir, plot_filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return beat1_apd, beat2_apd

    except Exception as e:
        print(f"--> Critical failure parsing {filename}: {str(e)}")
        return -1, -1


def batch_process_pipeline(data_folder, output_plot_dir, output_csv_path):
    """Loops through data folder, coordinates processing, and outputs matrix dataframe."""
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".npy")])

    if not files:
        print(f"No .npy files found in {data_folder}. Please check path.")
        return

    all_results = []
    print(f"Found {len(files)} traces to process...")

    for idx, filename in enumerate(files):
        full_filepath = os.path.join(data_folder, filename)
        beat1, beat2 = process_single_trace(full_filepath, filename, output_plot_dir)

        all_results.append({
            "Trace_Index": idx,
            "Filename": filename,
            "First_Beat_APD": beat1,
            "Second_Beat_APD": beat2
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"Progress: Completed {idx + 1}/{len(files)} files.")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n[Success] Data Matrix exported to: {output_csv_path}")
    print(f"[Success] 4-panel plots saved to: {output_plot_dir}")


if __name__ == "__main__":
    INPUT_FOLDER   = "data/2025-10-22/rec03/timeseries ratiometric raw/"
    PLOT_OUTPUT_DIR = "rec03_corrected_plots/"
    FINAL_CSV_PATH  = "batch_apd_reportrec03.csv"

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    batch_process_pipeline(INPUT_FOLDER, PLOT_OUTPUT_DIR, FINAL_CSV_PATH)


# STEP 2: CLUSTERING

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.stats import zscore
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── Settings ────────────────────────────────────────────────────────────────

recnum          = "rec07"
if recnum == "rec02" or "rec07":
    folder          = f"data/2025-07-23/{recnum}/timeseries ratiometric raw/"
    output_dir      = f"data/2025-07-23/{recnum}/"
elif recnum == "rec03" or "rec08":
    folder          = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
    output_dir      = f"data/2025-10-22/{recnum}/"
N_CLUSTERS      = 4
sampling_rate   = 250.0
MIN_APD_MS      = 140.0
MAX_APD_MS      = 600.0
MAX_PHASE_MS    = 500.0
APD90_THRESHOLD = 0.1    # signal has fallen to 10% of peak = APD90
APD25_THRESHOLD = 0.75   # signal has fallen to 75% of peak = APD25
OUTLIER_ZSCORE  = 3.0

files    = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
n_traces = len(files)
print(f"Found {n_traces} traces in {folder}")


# ─── Filter helpers ───────────────────────────────────────────────────────────

def butter_lowpass(data, cutoff=15.0, fs=250.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)

def get_cutoff(raw_norm, sampling_rate):
    n         = len(raw_norm)
    fft_vals  = np.abs(fft(raw_norm))
    fft_freqs = fftfreq(n, d=1.0 / sampling_rate)
    half      = n // 2
    freqs_s   = fft_freqs[:half]
    vals_s    = fft_vals[:half]
    slope     = np.gradient(vals_s)
    i0        = np.searchsorted(freqs_s, 3.0)
    i1        = np.searchsorted(freqs_s, 50.0)
    idx       = i0 + np.argmax(slope[i0:i1])
    return float(np.clip(freqs_s[idx], 15.0, 45.0))


# ─── APD threshold crossing — always searched AFTER peak ─────────────────────

def find_apd_crossing(beat_norm, threshold):
    """
    Find the first sample AFTER the peak where the signal falls below threshold.
    Returns the index relative to the start of beat_norm, or None if not found.
    This avoids catching the upstroke crossing the same threshold on the way up.
    """
    peak_idx = int(np.argmax(beat_norm))
    # only search the downslope — everything after the peak
    after_peak = beat_norm[peak_idx:]
    below = np.where(after_peak < threshold)[0]
    if len(below) == 0:
        return None
    return peak_idx + below[0]


# ─── Notch detection (adapted from ActionPytential paper) ────────────────────

def detect_notch(beat_norm, spm):
    beat_len  = len(beat_norm)
    peak_idx  = int(np.argmax(beat_norm))
    third_end = peak_idx + int((beat_len - peak_idx) / 3)
    third_end = min(third_end, beat_len - 1)

    if third_end <= peak_idx + 5:
        return 0.0, 0.0

    segment = beat_norm[peak_idx:third_end]
    x       = np.linspace(0, 1, len(segment))

    best_notch_score = 0.0
    notch_depth      = 0.0

    for degree in range(5, 16):
        if degree >= len(segment):
            break
        try:
            coeffs     = np.polyfit(x, segment, degree)
            poly_deriv = np.polyder(np.poly1d(coeffs))
            deriv_vals = poly_deriv(x)
            pos_trend  = float(np.max(deriv_vals[2:]))
            if pos_trend > best_notch_score:
                best_notch_score = pos_trend
                notch_depth      = float(1.0 - np.min(segment))
        except Exception:
            continue

    return best_notch_score, notch_depth


# ─── Core pipeline ────────────────────────────────────────────────────────────

def process_trace(raw_trace):
    n       = len(raw_trace)
    time_ms = np.linspace(0, 1000, n)
    spm     = n / 1000.0

    rng = raw_trace.max() - raw_trace.min()
    if rng < 1e-9:
        return {"success": False, "reason": "flat"}

    raw_norm = (raw_trace - raw_trace.min()) / rng
    cutoff   = get_cutoff(raw_norm, sampling_rate)
    smooth   = butter_lowpass(raw_norm, cutoff=cutoff, fs=sampling_rate)
    fd       = np.gradient(smooth)
    sd       = np.gradient(fd)

    min_dist = int(100 * spm)
    ups, _   = find_peaks(fd, height=0.05, distance=min_dist, prominence=0.05)

    if len(ups) == 0:
        return {"success": False, "reason": "no_upstroke"}

    min_apd_f   = int(MIN_APD_MS * spm)
    max_phase_f = int(MAX_PHASE_MS * spm)
    steeps, repols = [], []

    for i, up_idx in enumerate(ups):
        next_up   = ups[i + 1] if i + 1 < len(ups) else n - 1
        win_start = up_idx + min_apd_f
        win_end   = min(up_idx + max_phase_f, next_up, n - 1)
        if win_start >= win_end:
            win_start = up_idx + 5
        steep = win_start + np.argmin(fd[win_start:win_end])
        steeps.append(steep)
        s_end = min(steep + 25, win_end)
        repol = steep + (np.argmax(sd[steep:s_end]) if s_end > steep else 0)
        repols.append(repol)

    # baseline correction
    bl = raw_norm.copy()
    for up, re in zip(ups, repols):
        bl[up:re] = np.nan
    x_all     = np.arange(n)
    x_valid   = x_all[~np.isnan(bl)]
    y_valid   = bl[~np.isnan(bl)]
    bl_interp = np.interp(x_all, x_valid, y_valid)
    bl_smooth = savgol_filter(bl_interp, window_length=21, polyorder=3)
    corrected = raw_norm - bl_smooth

    # APD90 and APD25 — both searched after peak only
    apd90_values = []
    apd25_values = []

    for up_idx in ups:
        beat_end  = int(min(up_idx + MAX_PHASE_MS * spm, n - 1))
        beat      = corrected[up_idx:beat_end]
        peak_val  = beat.max()
        if peak_val <= 0:
            continue
        beat_norm = beat / peak_val

        idx90 = find_apd_crossing(beat_norm, APD90_THRESHOLD)
        if idx90 is not None:
            apd90_values.append(idx90 / spm)

        idx25 = find_apd_crossing(beat_norm, APD25_THRESHOLD)
        if idx25 is not None:
            apd25_values.append(idx25 / spm)

    apd90_ms = float(np.mean(apd90_values)) if apd90_values else np.nan
    apd25_ms = float(np.mean(apd25_values)) if apd25_values else np.nan

    return {
        "success":          True,
        "raw_norm":         raw_norm,
        "smooth":           smooth,
        "first_deriv":      fd,
        "second_deriv":     sd,
        "corrected":        corrected,
        "time_ms":          time_ms,
        "upstroke_indices": ups,
        "steep_indices":    np.array(steeps),
        "repol_indices":    np.array(repols),
        "apd90_ms":         apd90_ms,
        "apd25_ms":         apd25_ms,
        "n_upstrokes":      len(ups),
    }


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(proc):
    corrected = proc["corrected"]
    ups       = proc["upstroke_indices"]
    n         = len(corrected)
    spm       = n / 1000.0

    up_idx   = ups[0]
    beat_end = int(min(up_idx + MAX_PHASE_MS * spm, n - 1))
    beat     = corrected[up_idx:beat_end]

    if len(beat) < 10:
        return None, None

    peak = beat.max()
    if peak <= 0:
        return None, None

    beat_norm = beat / peak
    beat_len  = len(beat_norm)

    apd90 = proc["apd90_ms"]
    apd25 = proc["apd25_ms"]

    # Feature 1: APD90
    # Feature 2: Triangulation (APD90 - APD25)
    # wide gap = long slow tail (LQT-like), narrow = fast repolarization
    if not np.isnan(apd90) and not np.isnan(apd25):
        triangulation = apd90 - apd25
    else:
        triangulation = 0.0

    # Feature 3 & 4: Notch score and depth
    notch_score, notch_depth = detect_notch(beat_norm, spm)

    # Feature 5: Post-peak minimum — negative = artifact/inverted trace
    peak_idx      = int(np.argmax(beat_norm))
    post_peak_min = float(beat_norm[peak_idx:].min())

    # Feature 6: Plateau noise — high = bumpy/noisy plateau
    p20           = int(0.20 * beat_len)
    p70           = int(0.70 * beat_len)
    plateau_noise = float(np.std(beat_norm[p20:p70]))

    features = {
        "apd90_ms":      apd90,
        "triangulation": triangulation,
        "notch_score":   notch_score,
        "notch_depth":   notch_depth,
        "post_peak_min": post_peak_min,
        "plateau_noise": plateau_noise,
    }

    vec = np.array([
        apd90 if not np.isnan(apd90) else -1.0,
        triangulation,
        notch_score,
        notch_depth,
        post_peak_min,
        plateau_noise,
    ], dtype=np.float32)

    return vec, features


# ─── Run over all traces ──────────────────────────────────────────────────────

print("Extracting features from all traces...")

feature_matrix = []
feature_dicts  = []
valid_indices  = []
failed_indices = []

for idx, fname in enumerate(files):
    if idx % 200 == 0:
        print(f"  {idx}/{n_traces}")

    raw  = np.load(os.path.join(folder, fname))
    proc = process_trace(raw)

    if not proc["success"]:
        failed_indices.append(idx)
        continue

    vec, feat = extract_features(proc)
    if vec is None:
        failed_indices.append(idx)
        continue

    feature_matrix.append(vec)
    feature_dicts.append(feat)
    valid_indices.append(idx)

feature_matrix = np.array(feature_matrix)
valid_indices  = np.array(valid_indices)

print(f"\nSuccessful: {len(valid_indices)}/{n_traces}")
print(f"Failed:     {len(failed_indices)}/{n_traces}")


# ─── Remove outliers ─────────────────────────────────────────────────────────

z_scores     = np.abs(zscore(feature_matrix, axis=0))
outlier_mask = (z_scores > OUTLIER_ZSCORE).any(axis=1)

print(f"Removing {outlier_mask.sum()} outlier traces (z-score > {OUTLIER_ZSCORE})")
print(f"Outlier trace indices: {valid_indices[outlier_mask]}")

feature_matrix_clean = feature_matrix[~outlier_mask]
valid_indices_clean  = valid_indices[~outlier_mask]

print(f"Traces remaining for clustering: {len(valid_indices_clean)}")


# ─── Elbow plot ───────────────────────────────────────────────────────────────

print("Computing elbow plot...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix_clean)

inertias = []
k_range  = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(k_range), inertias, "o-", color="black", linewidth=2)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow plot — choose k at the bend")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "elbow_plot.png"), dpi=200)
plt.show()


# ─── K-means clustering ───────────────────────────────────────────────────────

print(f"Clustering with k={N_CLUSTERS}...")
km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
labels = km.fit_predict(X_scaled)

all_labels = np.full(n_traces, -1, dtype=int)
all_labels[valid_indices_clean] = labels

np.save(os.path.join(output_dir, "cluster_labels.npy"), all_labels)
print("Saved cluster_labels.npy")


# ─── PCA visualisation ────────────────────────────────────────────────────────

pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_scaled)
colors = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))

plt.figure(figsize=(8, 6))
for c in range(N_CLUSTERS):
    mask = labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[c], label=f"Cluster {c}  (n={mask.sum()})",
                s=8, alpha=0.6)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA of trace features — coloured by cluster")
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_clusters.png"), dpi=200)
plt.show()


# ─── Example traces per cluster ───────────────────────────────────────────────

print("Plotting example traces per cluster...")
N_EXAMPLES = 6

for c in range(N_CLUSTERS):
    cluster_trace_indices = valid_indices_clean[labels == c]
    sample = cluster_trace_indices[:N_EXAMPLES]

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 3),
                             sharey=True)
    if len(sample) == 1:
        axes = [axes]

    for ax, tidx in zip(axes, sample):
        raw  = np.load(os.path.join(folder, files[tidx]))
        proc = process_trace(raw)
        if proc["success"]:
            ax.plot(proc["time_ms"], proc["corrected"], color="black", linewidth=1)
            ups = proc["upstroke_indices"]
            if len(ups) > 0:
                up_idx    = ups[0]
                spm_plot  = len(proc["corrected"]) / 1000.0
                beat_end  = int(min(up_idx + MAX_PHASE_MS * spm_plot,
                                    len(proc["corrected"]) - 1))
                beat      = proc["corrected"][up_idx:beat_end]
                peak_val  = beat.max()
                if peak_val > 0:
                    beat_norm = beat / peak_val

                    # APD90 — after peak only
                    idx90 = find_apd_crossing(beat_norm, APD90_THRESHOLD)
                    if idx90 is not None:
                        ax.axvline(proc["time_ms"][up_idx + idx90],
                                   color="red", linewidth=1, linestyle="--")

                    # APD25 — after peak only
                    idx25 = find_apd_crossing(beat_norm, APD25_THRESHOLD)
                    if idx25 is not None:
                        ax.axvline(proc["time_ms"][up_idx + idx25],
                                   color="blue", linewidth=1, linestyle="--")

        ax.set_title(f"#{tidx}", fontsize=8)
        ax.set_xlabel("Time [ms]", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Corrected Signal", fontsize=8)
    fig.suptitle(f"Cluster {c}  —  {len(cluster_trace_indices)} traces total  "
                 f"(red=APD90, blue=APD25)", fontsize=9)
    plt.tight_layout()
    fname_out = os.path.join(output_dir, f"cluster_{c}_examples.png")
    plt.savefig(fname_out, dpi=200)
    plt.show()
    print(f"  Saved {fname_out}")


# ─── Feature summary per cluster ──────────────────────────────────────────────

feature_names = [
    "apd90_ms", "triangulation", "notch_score",
    "notch_depth", "post_peak_min", "plateau_noise"
]

print("\n── Mean feature values per cluster ─────────────────────────────")
header = f"{'Feature':<20}" + "".join(f"  C{c:>6}" for c in range(N_CLUSTERS))
print(header)
for fi, name in enumerate(feature_names):
    row = f"{name:<20}"
    for c in range(N_CLUSTERS):
        mask = labels == c
        val  = feature_matrix_clean[mask, fi].mean()
        row += f"  {val:>7.3f}"
    print(row)


# ─── Save ─────────────────────────────────────────────────────────────────────

np.save(os.path.join(output_dir, "feature_matrix.npy"), feature_matrix_clean)
np.save(os.path.join(output_dir, "valid_indices.npy"),  valid_indices_clean)
print("\nSaved feature_matrix.npy and valid_indices.npy")
print("Done.")