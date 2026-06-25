# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
# from scipy.fft import fft, fftfreq
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import warnings
# warnings.filterwarnings("ignore")

# # ─── Settings ───────────────────────────────────────────────────────────────

# recnum         = "rec02"
# folder         = f"data/2025-07-23/{recnum}/timeseries ratiometric raw/"
# output_dir     = f"data/2025-07-23/{recnum}/"
# N_CLUSTERS     = 6       # adjust after seeing elbow plot
# sampling_rate  = 250.0
# MIN_APD_MS     = 140.0
# MAX_APD_MS     = 600.0
# MAX_PHASE_MS   = 500.0
# APD_THRESHOLD  = 0.1     # for APD90 on corrected trace

# files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
# n_traces = len(files)
# print(f"Found {n_traces} traces in {folder}")


# # ─── Filter helpers ──────────────────────────────────────────────────────────

# def butter_lowpass(data, cutoff=15.0, fs=250.0, order=4):
#     nyq = 0.5 * fs
#     b, a = butter(order, cutoff / nyq, btype="low", analog=False)
#     return filtfilt(b, a, data)

# def get_cutoff(raw_norm, sampling_rate):
#     n = len(raw_norm)
#     fft_vals  = np.abs(fft(raw_norm))
#     fft_freqs = fftfreq(n, d=1.0 / sampling_rate)
#     half      = n // 2
#     freqs_s   = fft_freqs[:half]
#     vals_s    = fft_vals[:half]
#     slope     = np.gradient(vals_s)
#     i0 = np.searchsorted(freqs_s, 3.0)
#     i1 = np.searchsorted(freqs_s, 50.0)
#     idx = i0 + np.argmax(slope[i0:i1])
#     return float(np.clip(freqs_s[idx], 15.0, 45.0))


# # ─── Core pipeline: returns corrected trace + all detection info ─────────────

# def process_trace(raw_trace):
#     """
#     Returns a dict with:
#       - corrected   : baseline-corrected trace (panel 4 equivalent)
#       - raw_norm    : normalized raw trace
#       - smooth      : butterworth filtered trace
#       - first_deriv, second_deriv
#       - upstroke_indices, steep_indices, repol_indices  (from panel 1 detection)
#       - apd_ms      : APD90 measured on corrected trace (panel 4)
#       - success     : bool
#     """
#     n       = len(raw_trace)
#     time_ms = np.linspace(0, 1000, n)
#     spm     = n / 1000.0

#     rng = raw_trace.max() - raw_trace.min()
#     if rng < 1e-9:
#         return {"success": False, "reason": "flat"}

#     raw_norm = (raw_trace - raw_trace.min()) / rng

#     cutoff = get_cutoff(raw_norm, sampling_rate)
#     smooth = butter_lowpass(raw_norm, cutoff=cutoff, fs=sampling_rate)
#     fd     = np.gradient(smooth)
#     sd     = np.gradient(fd)

#     min_dist = int(100 * spm)
#     ups, _   = find_peaks(fd, height=0.05, distance=min_dist, prominence=0.05)

#     if len(ups) == 0:
#         return {"success": False, "reason": "no_upstroke"}

#     min_apd_f   = int(MIN_APD_MS * spm)
#     max_phase_f = int(MAX_PHASE_MS * spm)
#     steeps, repols = [], []

#     for i, up_idx in enumerate(ups):
#         next_up      = ups[i + 1] if i + 1 < len(ups) else n - 1
#         win_start    = up_idx + min_apd_f
#         win_end      = min(up_idx + max_phase_f, next_up, n - 1)
#         if win_start >= win_end:
#             win_start = up_idx + 5
#         steep = win_start + np.argmin(fd[win_start:win_end])
#         steeps.append(steep)
#         s_end = min(steep + 25, win_end)
#         repol = steep + (np.argmax(sd[steep:s_end]) if s_end > steep else 0)
#         repols.append(repol)

#     # ── Baseline correction (same as panel 3→4) ──────────────────────────
#     bl = raw_norm.copy()
#     for up, re in zip(ups, repols):
#         bl[up:re] = np.nan
#     x_all   = np.arange(n)
#     x_valid = x_all[~np.isnan(bl)]
#     y_valid = bl[~np.isnan(bl)]
#     bl_interp  = np.interp(x_all, x_valid, y_valid)
#     bl_smooth  = savgol_filter(bl_interp, window_length=21, polyorder=3)
#     corrected  = raw_norm - bl_smooth

#     # ── APD90 on corrected trace ──────────────────────────────────────────
#     apd_values = []
#     for up_idx in ups:
#         beat_start = up_idx
#         beat_end   = int(min(up_idx + MAX_PHASE_MS * spm, n - 1))
#         beat       = corrected[beat_start:beat_end]
#         peak       = beat.max()
#         if peak <= 0:
#             continue
#         beat_norm = beat / peak
#         below = np.where(beat_norm[1:] < APD_THRESHOLD)[0]
#         if len(below) == 0:
#             continue
#         repol_idx = beat_start + below[0] + 1
#         apd_values.append((repol_idx - up_idx) / spm)

#     apd_ms = float(np.mean(apd_values)) if apd_values else np.nan

#     return {
#         "success":          True,
#         "raw_norm":         raw_norm,
#         "smooth":           smooth,
#         "first_deriv":      fd,
#         "second_deriv":     sd,
#         "corrected":        corrected,
#         "time_ms":          time_ms,
#         "upstroke_indices": ups,
#         "steep_indices":    np.array(steeps),
#         "repol_indices":    np.array(repols),
#         "apd_ms":           apd_ms,
#         "n_upstrokes":      len(ups),
#     }


# # ─── Feature extraction from corrected trace ─────────────────────────────────

# def extract_features(proc):
#     corrected = proc["corrected"]
#     time_ms   = proc["time_ms"]
#     ups       = proc["upstroke_indices"]
#     n         = len(corrected)
#     spm       = n / 1000.0

#     up_idx   = ups[0]
#     beat_end = int(min(up_idx + MAX_PHASE_MS * spm, n - 1))
#     beat     = corrected[up_idx:beat_end]

#     if len(beat) < 10:
#         return None, None

#     peak = beat.max()
#     if peak <= 0:
#         return None, None

#     beat_norm = beat / peak
#     beat_len  = len(beat_norm)

#     apd = proc["apd_ms"]

#     # ── existing features ────────────────────────────────────────────────
#     upstroke_steepness = float(proc["first_deriv"][up_idx])

#     p20 = int(0.20 * beat_len)
#     p70 = int(0.70 * beat_len)
#     plateau_std  = float(np.std(beat_norm[p20:p70]))
#     plateau_mean = float(np.mean(beat_norm[p20:p70]))

#     peak_idx      = int(np.argmax(beat_norm))
#     post_peak_min = float(beat_norm[peak_idx:].min())

#     p70_abs = up_idx + p70
#     p95_abs = min(up_idx + int(0.95 * beat_len), n - 1)
#     repol_slope = float(np.mean(proc["first_deriv"][p70_abs:p95_abs])) \
#                   if p95_abs > p70_abs else 0.0

#     sd_plateau     = float(np.std(proc["second_deriv"][up_idx + p20: up_idx + p70]))
#     beat_consistency = 1.0 if proc["n_upstrokes"] >= 2 else 0.0

#     if proc["n_upstrokes"] >= 2:
#         up2       = ups[1]
#         gap_start = proc["repol_indices"][0] + 5
#         gap_end   = up2 - 5
#         baseline_noise = float(np.std(corrected[gap_start:gap_end])) \
#                          if gap_end > gap_start else 0.0
#     else:
#         baseline_noise = float(np.std(corrected[:up_idx - 5])) if up_idx > 10 else 0.0

#     # ── NEW: early notch score ───────────────────────────────────────────
#     # Look at first 25% of the beat for a local dip in the first derivative
#     # A notch = the signal slows down briefly just after the upstroke peak
#     p25 = int(0.25 * beat_len)
#     early_fd = proc["first_deriv"][up_idx: up_idx + p25]
#     # score = how much the derivative dips negative in the early window
#     # zero if no dip, positive number if there's a notch
#     early_notch_score = float(max(0.0, -early_fd.min()))

#     # ── NEW: repolarization duration (direct) ────────────────────────────
#     # Time from peak to 10% of peak on the way down, in ms
#     # This directly separates long repol from short repol
#     peak_abs = up_idx + peak_idx
#     after_peak = beat_norm[peak_idx:]
#     below10 = np.where(after_peak < 0.10)[0]
#     if len(below10) > 0:
#         repol_duration_ms = float(below10[0] / spm)
#     else:
#         repol_duration_ms = float(len(after_peak) / spm)  # didn't reach 10%

#     # ── NEW: plateau flatness in early vs late repol ─────────────────────
#     # ratio of signal level at 30% vs 70% of beat
#     # high ratio = flat plateau (signal stays high for long)
#     # low ratio  = triangular (signal drops steadily)
#     idx30 = int(0.30 * beat_len)
#     idx70 = int(0.70 * beat_len)
#     level_30 = float(beat_norm[idx30]) if idx30 < beat_len else 0.0
#     level_70 = float(beat_norm[idx70]) if idx70 < beat_len else 0.0
#     plateau_ratio = level_30 / (level_70 + 1e-6)

#     features = {
#         "apd_ms":             apd,
#         "upstroke_steepness": upstroke_steepness,
#         "plateau_std":        plateau_std,
#         "plateau_mean":       plateau_mean,
#         "post_peak_min":      post_peak_min,
#         "repol_slope":        repol_slope,
#         "secondary_hump":     sd_plateau,
#         "beat_consistency":   beat_consistency,
#         "baseline_noise":     baseline_noise,
#         "early_notch_score":  early_notch_score,   # NEW
#         "repol_duration_ms":  repol_duration_ms,   # NEW
#         "plateau_ratio":      plateau_ratio,        # NEW
#     }

#     vec = np.array([
#         apd if not np.isnan(apd) else -1.0,
#         upstroke_steepness,
#         plateau_std,
#         plateau_mean,
#         post_peak_min,
#         repol_slope,
#         sd_plateau,
#         beat_consistency,
#         baseline_noise,
#         early_notch_score,    # NEW
#         repol_duration_ms,    # NEW
#         plateau_ratio,        # NEW
#     ], dtype=np.float32)

#     return vec, features

# # ─── Run over all traces ─────────────────────────────────────────────────────

# print("Extracting features from all traces...")

# feature_matrix = []
# feature_dicts  = []
# valid_indices  = []   # trace indices that succeeded
# failed_indices = []

# for idx, fname in enumerate(files):
#     if idx % 200 == 0:
#         print(f"  {idx}/{n_traces}")

#     raw = np.load(os.path.join(folder, fname))
#     proc = process_trace(raw)

#     if not proc["success"]:
#         failed_indices.append(idx)
#         continue

#     vec, feat = extract_features(proc)
#     if vec is None:
#         failed_indices.append(idx)
#         continue

#     feature_matrix.append(vec)
#     feature_dicts.append(feat)
#     valid_indices.append(idx)

# feature_matrix = np.array(feature_matrix)
# valid_indices  = np.array(valid_indices)

# print(f"\nSuccessful: {len(valid_indices)}/{n_traces}")
# print(f"Failed:     {len(failed_indices)}/{n_traces}")


# # ─── Elbow plot to pick K ────────────────────────────────────────────────────

# print("Computing elbow plot...")
# scaler    = StandardScaler()
# X_scaled  = scaler.fit_transform(feature_matrix)

# inertias = []
# k_range  = range(2, 12)
# for k in k_range:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(X_scaled)
#     inertias.append(km.inertia_)

# plt.figure(figsize=(7, 4))
# plt.plot(list(k_range), inertias, "o-", color="black", linewidth=2)
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia (within-cluster sum of squares)")
# plt.title("Elbow plot — choose k at the bend")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "elbow_plot.png"), dpi=200)
# plt.show()


# # ─── K-means clustering ──────────────────────────────────────────────────────

# print(f"Clustering with k={N_CLUSTERS}...")
# km       = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
# labels   = km.fit_predict(X_scaled)

# # full label array (failed traces get label = -1)
# all_labels = np.full(n_traces, -1, dtype=int)
# all_labels[valid_indices] = labels

# np.save(os.path.join(output_dir, "cluster_labels.npy"), all_labels)
# print(f"Saved cluster_labels.npy")


# # ─── PCA visualisation of clusters ───────────────────────────────────────────

# pca    = PCA(n_components=2)
# X_pca  = pca.fit_transform(X_scaled)

# colors = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))
# plt.figure(figsize=(8, 6))
# for c in range(N_CLUSTERS):
#     mask = labels == c
#     plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
#                 color=colors[c], label=f"Cluster {c}  (n={mask.sum()})",
#                 s=8, alpha=0.6)
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
# plt.title("PCA of trace features — coloured by cluster")
# plt.legend(markerscale=3)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "pca_clusters.png"), dpi=200)
# plt.show()


# # ─── Plot 6 example traces per cluster ───────────────────────────────────────

# print("Plotting example traces per cluster...")

# N_EXAMPLES = 6

# for c in range(N_CLUSTERS):
#     cluster_trace_indices = valid_indices[labels == c]
#     sample = cluster_trace_indices[:N_EXAMPLES]   # first N examples

#     fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 3),
#                              sharey=True)
#     if len(sample) == 1:
#         axes = [axes]

#     for ax, tidx in zip(axes, sample):
#         raw  = np.load(os.path.join(folder, files[tidx]))
#         proc = process_trace(raw)
#         if proc["success"]:
#             ax.plot(proc["time_ms"], proc["corrected"], color="black", linewidth=1)
#         ax.set_title(f"#{tidx}", fontsize=8)
#         ax.set_xlabel("Time [ms]", fontsize=7)
#         ax.tick_params(labelsize=7)

#     axes[0].set_ylabel("Corrected Signal", fontsize=8)
#     fig.suptitle(f"Cluster {c}  —  {len(cluster_trace_indices)} traces total",
#                  fontsize=10)
#     plt.tight_layout()
#     fname_out = os.path.join(output_dir, f"cluster_{c}_examples.png")
#     plt.savefig(fname_out, dpi=200)
#     plt.show()
#     print(f"  Saved {fname_out}")


# # ─── Feature summary per cluster ────────────────────────────────────────────

# feature_names = [
#     "apd_ms", "upstroke_steepness", "plateau_std", "plateau_mean",
#     "post_peak_min", "repol_slope", "secondary_hump",
#     "beat_consistency", "baseline_noise",
#     "early_notch_score", "repol_duration_ms", "plateau_ratio"
# ]

# print("\n── Mean feature values per cluster ──────────────────────────────")
# header = f"{'Feature':<22}" + "".join(f"  C{c:>5}" for c in range(N_CLUSTERS))
# print(header)
# for fi, name in enumerate(feature_names):
#     row = f"{name:<22}"
#     for c in range(N_CLUSTERS):
#         mask = labels == c
#         val  = feature_matrix[mask, fi].mean()
#         row += f"  {val:>6.3f}"
#     print(row)


# # ─── Save feature matrix and valid indices ───────────────────────────────────

# np.save(os.path.join(output_dir, "feature_matrix.npy"),  feature_matrix)
# np.save(os.path.join(output_dir, "valid_indices.npy"),    valid_indices)
# print("\nSaved feature_matrix.npy and valid_indices.npy")
# print("Done.")




# REC07777

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
folder          = f"data/2025-07-23/{recnum}/timeseries ratiometric raw/"
output_dir      = f"data/2025-07-23/{recnum}/"
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