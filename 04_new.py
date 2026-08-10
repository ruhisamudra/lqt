# STEP 1: GENERATE CLUSTER LABELS AND TRIANGLES FOR BASELINE AND LQT
import numpy as np
import os
import vtk

from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import HDBSCAN

# Try importing UMAP for better density-based clustering; fall back to PCA if unavailable
try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    HAS_UMAP = False

# ── Settings ──────────────────────────────────────────────────────────────
BASELINE_RECNUM = "rec02"
LQT_RECNUM = "rec07"

ALIGN_WINDOW_MS = 400.0     
RESAMPLE_POINTS = 100       

# Parameters tuned to reduce noise and allow spatial cohesion testing
MIN_CLUSTER_SIZE = 30     
MIN_SAMPLES = 2           

# ── Feature Extraction ──────────────────────────────────────────────────

def extract_trace_features(vec):
    """
    Extracts shape features that capture action potential dynamics.
    """
    max_dvdt = np.max(np.gradient(vec))
    
    # Version-safe integration for NumPy 2.0+ and 1.x
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', np.sum))
    auc_repol = trapz_func(vec[15:70])
    
    half_max_width = np.sum(vec > 0.5)
    ead_prominence = np.max(vec[30:80]) if len(vec) > 80 else 0.0
    decay_slope = np.min(np.gradient(vec[10:50]))

    return [max_dvdt, auc_repol, half_max_width, ead_prominence, decay_slope]

def get_folder(recnum):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    return f"data/{date_subfolder}/{recnum}/timeseries corrected new/"

def get_mesh_path(recnum):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    return f"data/{date_subfolder}/{recnum}/mesh/mesh0.gltf"

def load_and_align_trace(filepath, sampling_rate=250.0):
    trace = np.load(filepath)
    n_points = len(trace)
    ms_per_frame = 1000.0 / sampling_rate

    trace_smooth = savgol_filter(trace, window_length=11, polyorder=3)
    t_min, t_max = trace_smooth.min(), trace_smooth.max()
    if t_max - t_min == 0:
        return None
    trace_norm = (trace_smooth - t_min) / (t_max - t_min)

    first_deriv = np.gradient(trace_norm)
    min_distance = int(100 / ms_per_frame)
    upstroke_indices, _ = find_peaks(first_deriv, height=0.05, distance=min_distance, prominence=0.05)
    if len(upstroke_indices) == 0:
        return None

    up_idx = upstroke_indices[0]  
    window_frames = int(ALIGN_WINDOW_MS / ms_per_frame)
    window_end = min(up_idx + window_frames, n_points)
    if window_end - up_idx < int(50 / ms_per_frame):
        return None

    segment = trace_norm[up_idx:window_end]
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, RESAMPLE_POINTS)
    return np.interp(x_new, x_old, segment)

def build_trace_matrix(recnum):
    folder = get_folder(recnum)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

    feature_vectors, tri_indices = [], []
    for i, fname in enumerate(files):
        vec = load_and_align_trace(os.path.join(folder, fname))
        if vec is not None:
            feats = extract_trace_features(vec)
            feature_vectors.append(feats)
            tri_indices.append(i)

    print(f"  {recnum}: {len(feature_vectors)} / {len(files)} traces usable "
          f"({len(feature_vectors) / len(files):.1%} coverage)")
    return np.array(feature_vectors), np.array(tri_indices)

def build_adjacency(recnum):
    reader = vtk.vtkGLTFImporter()
    reader.SetFileName(get_mesh_path(recnum))
    reader.Update()
    r = reader.GetRenderer()
    actors = r.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextActor()
    polydata = actor.GetMapper().GetInput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    mesh = triangle_filter.GetOutput()
    num_triangles = mesh.GetNumberOfCells()

    vertex_to_triangles = {}
    for i in range(num_triangles):
        cell = mesh.GetCell(i)
        for v in range(3):
            vid = cell.GetPointId(v)
            vertex_to_triangles.setdefault(vid, []).append(i)

    neighbors = {}
    for i in range(num_triangles):
        cell = mesh.GetCell(i)
        neighbor_set = set()
        for v in range(3):
            vid = cell.GetPointId(v)
            neighbor_set.update(vertex_to_triangles[vid])
        neighbor_set.discard(i)
        neighbors[i] = [int(n) for n in neighbor_set]

    return neighbors, num_triangles

def spatial_cohesion(cluster_labels, tri_indices, neighbors):
    """
    Evaluates whether clustered triangles form contiguous spatial domains.
    Includes ALL triangles in neighbor-matching to account for sparse mesh boundaries.
    """
    # Build complete mapping for all indices (including noise as -1)
    tri_to_label = {int(t): int(c) for t, c in zip(tri_indices, cluster_labels)}
    
    # Evaluate cohesion specifically for non-noise triangles
    clustered_tris = [t for t, c in tri_to_label.items() if c != -1]
    if len(clustered_tris) < 10:
        print("  Not enough clustered (non-noise) triangles for a cohesion test.")
        return

    same_cluster_neighbor_fracs = []
    for t in clustered_tris:
        label = tri_to_label[t]
        t_neighbors = neighbors.get(t, [])
        # Find neighbors that exist in our dataset
        valid_neighs = [n for n in t_neighbors if n in tri_to_label]
        
        if valid_neighs:
            # Fraction of neighbors belonging to the SAME non-noise cluster
            matches = [tri_to_label[n] == label for n in valid_neighs]
            same_cluster_neighbor_fracs.append(np.mean(matches))

    if not same_cluster_neighbor_fracs:
        print("  No neighboring triangles found on mesh.")
        return

    observed = np.mean(same_cluster_neighbor_fracs)

    # Permutation test across mapped triangles
    labels_pool = [tri_to_label[t] for t in clustered_tris]
    rng = np.random.default_rng(0)
    shuffled_scores = []
    for _ in range(200):
        shuffled = rng.permutation(labels_pool)
        shuffled_map = dict(zip(clustered_tris, shuffled))
        # Keep noise labels fixed for fair baseline comparisons
        full_shuffled_map = {**tri_to_label, **shuffled_map}
        
        fracs = []
        for t in clustered_tris:
            label = full_shuffled_map[t]
            valid_neighs = [n for n in neighbors.get(t, []) if n in full_shuffled_map]
            if valid_neighs:
                fracs.append(np.mean([full_shuffled_map[n] == label for n in valid_neighs]))
        if fracs:
            shuffled_scores.append(np.mean(fracs))

    expected = np.mean(shuffled_scores) if shuffled_scores else 0.0

    print(f"  Observed same-cluster neighbor fraction: {observed:.3f}")
    print(f"  Expected under random spatial arrangement: {expected:.3f}")
    ratio = (observed / expected) if expected > 0 else 0.0
    print(f"  Cohesion ratio (observed/expected): {ratio:.2f}x "
          f"{'-- spatially congregating' if ratio > 1.2 else '-- weak/no spatial pattern'}")

def run_clustering():
    print(f"=== Morphology clustering: {BASELINE_RECNUM} (baseline) vs {LQT_RECNUM} (LQT) ===\n")

    print("Loading and extracting shape features...")
    X_base, tri_base = build_trace_matrix(BASELINE_RECNUM)
    X_lqt, tri_lqt = build_trace_matrix(LQT_RECNUM)

    X_all = np.vstack([X_base, X_lqt])
    condition_labels = np.array(['baseline'] * len(X_base) + ['lqt'] * len(X_lqt))

    print("\nScaling biological features...")
    X_scaled = RobustScaler().fit_transform(X_all)

    if HAS_UMAP:
        print("\nReducing feature space with UMAP...")
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        print("\nReducing feature space with PCA...")
        embedding = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    print("Clustering with HDBSCAN on Reduced Space...")
    clusterer = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        copy=True
    )

    cluster_labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    print(f"  Found {n_clusters} clusters, {n_noise} traces labeled as noise/unclustered "
          f"({n_noise / len(cluster_labels):.1%})")

    labels_base = cluster_labels[:len(X_base)]
    labels_lqt = cluster_labels[len(X_base):]

    print(f"\nCluster composition:")
    print(f"  {'Cluster':<10}{'Baseline count':<16}{'LQT count':<12}{'LQT enrichment'}")
    for c in sorted(set(cluster_labels)):
        n_b = np.sum(labels_base == c)
        n_l = np.sum(labels_lqt == c)
        total_b, total_l = len(labels_base), len(labels_lqt)
        frac_b = n_b / total_b if total_b else 0
        frac_l = n_l / total_l if total_l else 0
        enrichment = (frac_l / frac_b) if frac_b > 0 else float('inf')
        tag = "noise" if c == -1 else f"cluster {c}"
        print(f"  {tag:<10}{n_b:<16}{n_l:<12}{enrichment:.2f}x")

    print(f"\nSpatial cohesion -- baseline ({BASELINE_RECNUM}):")
    neighbors_base, _ = build_adjacency(BASELINE_RECNUM)
    spatial_cohesion(labels_base, tri_base, neighbors_base)

    print(f"\nSpatial cohesion -- LQT ({LQT_RECNUM}):")
    neighbors_lqt, _ = build_adjacency(LQT_RECNUM)
    spatial_cohesion(labels_lqt, tri_lqt, neighbors_lqt)

    os.makedirs("cluster_output", exist_ok=True)
    np.save(f"cluster_output/{BASELINE_RECNUM}_cluster_labels.npy", labels_base)
    np.save(f"cluster_output/{BASELINE_RECNUM}_cluster_triangles.npy", tri_base)
    np.save(f"cluster_output/{LQT_RECNUM}_cluster_labels.npy", labels_lqt)
    np.save(f"cluster_output/{LQT_RECNUM}_cluster_triangles.npy", tri_lqt)
    print(f"\nSaved per-triangle cluster assignments to cluster_output/")

if __name__ == '__main__':
    run_clustering()

# STEP 2: PLOT CLUSTERS ON HEART
import vtk
import vtk.util.numpy_support
import numpy as np
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force uniform Times style for all figures
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# ── CLUSTER COLOR PALETTE ────────────────────────────────────────────────────
# Fixed palette keyed by cluster ID -- same ID gets the same color in both the
# BASELINE (B) and LQT (L) views, since cluster IDs come from a single joint
# HDBSCAN fit across both conditions (see run_clustering.py). Noise (-1) is
# always gray.
CLUSTER_PALETTE = [
    (0.89, 0.10, 0.11),  # red
    (0.22, 0.49, 0.72),  # blue
    (0.30, 0.69, 0.29),  # green
    (1.00, 0.50, 0.00),  # orange
    (0.60, 0.31, 0.64),  # purple
    (0.90, 0.80, 0.10),  # yellow
    (0.65, 0.34, 0.16),  # brown
    (0.97, 0.51, 0.75),  # pink
    (0.00, 0.75, 0.75),  # teal
    (0.50, 0.50, 0.00),  # olive
]
NOISE_COLOR = (0.55, 0.55, 0.55)


def build_cluster_color_map(all_ids):
    """Assigns a fixed RGB color to each cluster ID. -1 (noise) is always gray."""
    non_noise = sorted(int(i) for i in set(all_ids) if int(i) != -1)
    color_map = {-1: NOISE_COLOR}
    for idx, cid in enumerate(non_noise):
        color_map[cid] = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
    return color_map


# ── ON-THE-FLY PLOT → vtkImageData ──────────────────────────────────────────

def generate_thumbnail_image(recnum, filenum):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    folder = f"data/{date_subfolder}/{recnum}/timeseries ratiometric raw/"
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

    if filenum < 0 or filenum >= len(files):
        return None

    try:
        raw_trace = np.load(os.path.join(folder, files[filenum]))
        n_points = len(raw_trace)
        time_ms = np.linspace(0, 1000, n_points)
        samples_per_ms = n_points / 1000.0
        sampling_rate = 250.0
        MIN_APD_MS = 140.0
        min_apd_frames = int(MIN_APD_MS * samples_per_ms)
        MAX_PHASE_WINDOW_MS = 600.0
        max_phase_frames = int(MAX_PHASE_WINDOW_MS * samples_per_ms)

        raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

        fft_vals = np.abs(fft(raw_norm))
        fft_freqs_arr = fftfreq(n_points, d=1.0 / sampling_rate)
        half_points = n_points // 2
        freqs_side = fft_freqs_arr[:half_points]
        vals_side = fft_vals[:half_points]
        fft_slope = np.gradient(vals_side)
        search_start_idx = np.searchsorted(freqs_side, 3.0)
        search_end_idx = np.searchsorted(freqs_side, 50.0)
        optimal_idx = search_start_idx + np.argmax(fft_slope[search_start_idx:search_end_idx])
        CUTOFF_FREQUENCY = float(np.clip(freqs_side[optimal_idx], 15.0, 45.0))

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            nyquist = 0.5 * fs
            b, a = butter(order, cutoff / nyquist, btype='low', analog=False)
            return filtfilt(b, a, data)

        raw_smooth_butter = butter_lowpass_filter(raw_norm, CUTOFF_FREQUENCY, sampling_rate)
        first_deriv_butter = np.gradient(raw_smooth_butter)
        second_deriv_butter = np.gradient(first_deriv_butter)

        min_distance_samples = int(100 * samples_per_ms)
        upstroke_indices, _ = find_peaks(
            first_deriv_butter, height=0.05,
            distance=min_distance_samples, prominence=0.05
        )

        steep_indices = []
        repol_indices = []
        for i, up_idx in enumerate(upstroke_indices):
            window_start = up_idx + min_apd_frames
            next_upstroke = upstroke_indices[i + 1] if i + 1 < len(upstroke_indices) else n_points - 1
            window_end = min(up_idx + max_phase_frames, next_upstroke, n_points - 1)
            if window_start >= window_end:
                window_start = up_idx + 5
            steep = window_start + np.argmin(first_deriv_butter[window_start:window_end])
            steep_indices.append(steep)
            search_end = min(steep + 25, window_end)
            repol = steep + (np.argmax(second_deriv_butter[steep:search_end]) if search_end > steep else 0)
            repol_indices.append(repol)

        baseline_trace = raw_norm.copy()
        for up, re in zip(upstroke_indices, repol_indices):
            baseline_trace[up:re] = np.nan
        x_all = np.arange(n_points)
        x_valid = x_all[~np.isnan(baseline_trace)]
        y_valid = baseline_trace[~np.isnan(baseline_trace)]
        if len(x_valid) < 2:
            return None
        baseline_interpolated = np.interp(x_all, x_valid, y_valid)
        baseline_smooth_savgol = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
        corrected_trace_savgol = raw_norm - baseline_smooth_savgol

        fig, axes = plt.subplots(4, 1, figsize=(4, 6), sharex=False)
        fig.patch.set_facecolor('white')
        axes[0].plot(time_ms, raw_norm, color="lightgray", linewidth=0.6)
        axes[0].plot(time_ms, raw_smooth_butter, color="black", linewidth=0.8)
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[0].axvline(time_ms[up], color="red", linewidth=1.0)
            axes[0].axvline(time_ms[st], color="magenta", linewidth=1.0)
            axes[0].axvline(time_ms[re], color="blue", linewidth=1.0)
        axes[0].set_ylabel("Signal", fontsize=6)
        axes[0].tick_params(labelsize=5)
        axes[1].plot(time_ms, first_deriv_butter, color="black", linewidth=0.8)
        axes[1].plot(time_ms, second_deriv_butter, color="gray", linewidth=0.6)
        axes[1].axhline(0, color="gray", linewidth=0.5)
        for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
            axes[1].axvline(time_ms[up], color="red", linewidth=1.0)
            axes[1].axvline(time_ms[st], color="magenta", linewidth=1.0)
            axes[1].axvline(time_ms[re], color="blue", linewidth=1.0)
        axes[1].set_ylabel("dF/dt", fontsize=6)
        axes[1].tick_params(labelsize=5)
        axes[2].plot(time_ms, raw_norm, color="black", linewidth=0.8)
        axes[2].plot(time_ms, baseline_smooth_savgol, color="green", linewidth=0.8)
        axes[2].set_ylabel("Baseline", fontsize=6)
        axes[2].tick_params(labelsize=5)
        axes[3].plot(time_ms, corrected_trace_savgol, color="black", linewidth=0.8)
        axes[3].set_ylabel("Corrected", fontsize=6)
        axes[3].set_xlabel("Time [ms]", fontsize=6)
        axes[3].tick_params(labelsize=5)

        fig.suptitle(f"Tri {filenum}", fontsize=7, y=1.01)
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)

        img_array = plt.imread(buf)
        img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]

        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(w, h, 1)
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

        flat = img_uint8[::-1, :, :].reshape(-1)
        vtk_arr = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_arr.SetNumberOfComponents(4)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        return vtk_img
    except Exception as e:
        print(f"--> Thumbnail generation failed for triangle {filenum}: {e}")
        return None


# ── APD COMPUTATION (corner readout only) ───────────────────────────────────

def compute_apd(recnum, triangle_index):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    folder = f"data/{date_subfolder}/{recnum}/timeseries corrected new/"
    prefix = "trace_corrected_triangle_"

    filepath = os.path.join(folder, f"{prefix}{triangle_index:04}.npy")
    if not os.path.exists(filepath):
        return None, None

    trace = np.load(filepath)
    trace_SG = savgol_filter(trace, window_length=21, polyorder=9)
    trace_mn = np.convolve(trace, np.ones(5) / 5, mode='same')

    baseline = np.mean(trace_SG[60:120])
    peak = np.max(trace_mn)
    if peak - baseline == 0:
        return None, None

    trace_SG_norm = (trace_SG - baseline) / (peak - baseline)
    trace_mn_norm = (trace_mn - baseline) / (peak - baseline)

    APD_threshold = 0.1
    up_crossings = np.where(
        (trace_mn_norm[:-1] < APD_threshold) & (trace_mn_norm[1:] >= APD_threshold) &
        (np.abs(trace_mn_norm[:-1] - trace_mn_norm[1:]) > 0.05)
    )[0]
    down_crossings = np.where((trace_SG_norm[:-1] >= APD_threshold) & (trace_SG_norm[1:] < APD_threshold))[0]

    if len(up_crossings) == 2 and len(down_crossings) == 2:
        apd1 = (down_crossings[0] - up_crossings[0]) * 4.0
        apd2 = (down_crossings[1] - up_crossings[1]) * 4.0
        return apd1, apd2
    return None, None


# ── CLUSTER DATA LOADING ─────────────────────────────────────────────────────

def load_cluster_labels(recnum):
    """Loads per-triangle cluster labels produced by run_clustering.py."""
    labels = np.load(f"cluster_output/{recnum}_cluster_labels.npy")
    tri_idx = np.load(f"cluster_output/{recnum}_cluster_triangles.npy")
    return labels, tri_idx


# ── MESH LOADING ──────────────────────────────────────────────────────────

def load_mesh(recnum):
    mesh_paths = {
        "rec02": "data/2025-07-23/rec02/mesh/",
        "rec07": "data/2025-07-23/rec07/mesh/",
        "rec03": "data/2025-10-22/rec03/mesh/",
        "rec08": "data/2025-10-22/rec08/mesh/"
    }
    mesh_path = mesh_paths.get(recnum, f"data/2025-10-22/{recnum}/mesh/")
    filename = mesh_path + "mesh0.gltf"
    reader = vtk.vtkGLTFImporter()
    reader.SetFileName(filename)
    reader.Update()

    r = reader.GetRenderer()
    actors = r.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextActor()
    polydata = None
    if isinstance(actor, vtk.vtkActor):
        mapper = actor.GetMapper()
        data = mapper.GetInput()
        if isinstance(data, vtk.vtkPolyData):
            polydata = data
    if polydata is None:
        raise RuntimeError(f"Could not load valid polydata from GLTF file for {recnum}.")

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    num_triangles = triangle_filter.GetOutput().GetNumberOfCells()

    triangle_centroids = []
    for i in range(num_triangles):
        cell = triangle_filter.GetOutput().GetCell(i)
        pts = cell.GetPoints()
        p0 = np.array(pts.GetPoint(0))
        p1 = np.array(pts.GetPoint(1))
        p2 = np.array(pts.GetPoint(2))
        triangle_centroids.append((p0 + p1 + p2) / 3)
    triangle_centroids = np.array(triangle_centroids)

    num_points = polydata.GetNumberOfPoints()
    center_of_mass = np.zeros(3)
    for i in range(num_points):
        pt = [0.0, 0.0, 0.0]
        polydata.GetPoint(i, pt)
        center_of_mass += pt
    center_of_mass /= num_points

    return polydata, triangle_filter, num_triangles, triangle_centroids, center_of_mass


def build_condition(recnum, label):
    """Loads mesh + cluster assignments + APD readout for one condition (B or L)."""
    print(f"Loading mesh + cluster data for {recnum} ({label})...")
    polydata, triangle_filter, num_triangles, triangle_centroids, center_of_mass = load_mesh(recnum)

    labels, tri_idx = load_cluster_labels(recnum)
    cluster_array = vtk.vtkIntArray()
    cluster_array.SetName("Cluster")
    cluster_array.SetNumberOfTuples(num_triangles)
    cluster_array.Fill(-1)
    for lbl, t in zip(labels, tri_idx):
        t = int(t)
        if 0 <= t < num_triangles:
            cluster_array.SetValue(t, int(lbl))
    triangle_filter.GetOutput().GetCellData().SetScalars(cluster_array)

    print(f"  Precomputing APD readout values for {recnum}...")
    apd1_list, apd2_list = [], []
    for i in range(num_triangles):
        a1, a2 = compute_apd(recnum, i)
        apd1_list.append(a1)
        apd2_list.append(a2)

    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(triangle_filter.GetOutput())
    cell_locator.BuildLocator()

    return {
        'recnum': recnum,
        'label': label,
        'polydata': polydata,
        'triangle_filter': triangle_filter,
        'num_triangles': num_triangles,
        'triangle_centroids': triangle_centroids,
        'center_of_mass': center_of_mass,
        'cluster_array': cluster_array,
        'apd1_list': apd1_list,
        'apd2_list': apd2_list,
        'cell_locator': cell_locator,
    }


# ── MAIN CORE ENGINE ──────────────────────────────────────────────────────────

def run_wave_explorer():
    THUMB_FRAC_W = 0.3
    THUMB_FRAC_H = 0.7
    MARGIN = 16

    RECNUMS = {'B': 'rec02', 'L': 'rec07'}

    if RECNUMS == {'B': 'rec02', 'L': 'rec07'}:
        TITLES = {
            'B': 'BASELINE (rec02, 07/23/2025)',
            'L': 'LQT (rec07, 07/23/2025)',
        }
    elif RECNUMS == {'B': 'rec03', 'L': 'rec08'}:
        TITLES = {
                    'B': 'BASELINE (rec03, 10/22/2025)',
                    'L': 'LQT (rec08, 10/22/2025)',
                }

    print('=== Starting Integrated Cardiac Cluster Explorer ===')
    print('Controls: [B] = Baseline | [L] = LQT | [F] = Search Triangle | '
          '[Enter] = Snap to Center | [Arrows] = Move Blue Dot | [+/-] = Zoom')

    print("Determining shared cluster color palette...")
    all_labels = []
    for recnum in RECNUMS.values():
        labels, _ = load_cluster_labels(recnum)
        all_labels.extend(labels.tolist())
    color_map = build_cluster_color_map(all_labels)
    sorted_ids = sorted(color_map.keys())

    conditions = {}
    for key, recnum in RECNUMS.items():
        conditions[key] = build_condition(recnum, TITLES[key])

    # ── Categorical (indexed) lookup table shared by both conditions ──
    lut = vtk.vtkLookupTable()
    lut.IndexedLookupOn()
    lut.SetNumberOfTableValues(len(sorted_ids))
    for i, cid in enumerate(sorted_ids):
        r, g, b = color_map[cid]
        lut.SetTableValue(i, r, g, b, 1.0)
        lut.SetAnnotation(vtk.vtkVariant(int(cid)), "Noise" if cid == -1 else f"Cluster {cid}")
    lut.Build()

    state = {
        'active':        'B',
        'current_tri':   0,
        'keys_held':     set(),
        'number_buffer': None,
        'last_thumb_tri': -1,
        'needs_update':  False,
    }
    state['sphere_pos'] = conditions['B']['triangle_centroids'][0].copy()

    renderer = vtk.vtkRenderer()
    renderer.SetLayer(0)
    renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    renderer.SetBackground(1.0, 1.0, 1.0)

    overlay_renderer = vtk.vtkRenderer()
    overlay_renderer.SetLayer(1)
    overlay_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    overlay_renderer.InteractiveOff()
    overlay_renderer.SetBackgroundAlpha(0.0)

    for key, cond in conditions.items():
        mapper_APD = vtk.vtkPolyDataMapper()
        mapper_APD.SetInputData(cond['triangle_filter'].GetOutput())
        mapper_APD.SetScalarModeToUseCellData()
        mapper_APD.SetLookupTable(lut)
        mapper_APD.SetScalarRange(sorted_ids[0], sorted_ids[-1])
        actor_APD = vtk.vtkActor()
        actor_APD.SetMapper(mapper_APD)
        actor_APD.SetVisibility(1 if key == state['active'] else 0)
        renderer.AddActor(actor_APD)
        cond['actor_APD'] = actor_APD
        cond['mapper_APD'] = mapper_APD

        mapper_wf = vtk.vtkPolyDataMapper()
        mapper_wf.SetInputData(cond['polydata'])
        actor_wf = vtk.vtkActor()
        actor_wf.SetMapper(mapper_wf)
        actor_wf.GetProperty().SetRepresentationToWireframe()
        actor_wf.GetProperty().SetColor(0.85, 0.85, 0.85)
        actor_wf.GetProperty().SetLineWidth(1.0)
        actor_wf.SetVisibility(1 if key == state['active'] else 0)
        renderer.AddActor(actor_wf)
        cond['actor_wf'] = actor_wf

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetRadius(0.28)
    sphereSource.SetCenter(state['sphere_pos'])
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphereSource.GetOutputPort())
    actor_sphere = vtk.vtkActor()
    actor_sphere.SetMapper(sphere_mapper)
    actor_sphere.GetProperty().SetColor(0.0, 0.0, 0.7)
    renderer.AddActor(actor_sphere)

    corner_text = vtk.vtkTextActor()
    corner_text.GetTextProperty().SetFontSize(25)
    corner_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    corner_text.GetTextProperty().SetFontFamilyToTimes()
    corner_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    corner_text.SetPosition(0.02, 0.02)
    renderer.AddActor2D(corner_text)

    view_text = vtk.vtkTextActor()
    view_text.GetTextProperty().SetFontSize(25)
    view_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    view_text.GetTextProperty().SetFontFamilyToTimes()
    view_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    view_text.SetPosition(0.02, 0.92)
    view_text.SetInput("View: BASELINE  [B = Baseline | L = LQT]")
    renderer.AddActor2D(view_text)

    title_text = vtk.vtkTextActor()
    title_text.GetTextProperty().SetFontSize(30)
    title_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    title_text.GetTextProperty().SetFontFamilyToTimes()
    title_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    title_text.SetPosition(0.02, 0.94)
    title_text.SetInput(conditions['B']['label'])
    renderer.AddActor2D(title_text)

    for i, cid in enumerate(sorted_ids):
        r, g, b = color_map[cid]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        entry_text = vtk.vtkTextActor()
        entry_text.SetInput(f"\u25A0 {label}")
        entry_text.GetTextProperty().SetFontSize(16)
        entry_text.GetTextProperty().SetFontFamilyToTimes()
        entry_text.GetTextProperty().SetColor(r, g, b)
        entry_text.GetTextProperty().BoldOn()
        entry_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        entry_text.SetPosition(0.02, legend_y_start - i * legend_y_step)
        renderer.AddActor2D(entry_text)

    blank_img = vtk.vtkImageData()
    blank_img.SetDimensions(1, 1, 1)
    blank_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    blank_img.GetPointData().GetScalars().Fill(255)
    thumb_mapper = vtk.vtkImageMapper()
    thumb_mapper.SetInputData(blank_img)
    thumb_mapper.SetColorWindow(255)
    thumb_mapper.SetColorLevel(127.5)
    thumb_actor = vtk.vtkActor2D()
    thumb_actor.SetMapper(thumb_mapper)
    thumb_actor.GetProperty().SetOpacity(0.0)
    overlay_renderer.AddActor2D(thumb_actor)

    def active_cond():
        return conditions[state['active']]

    def refresh_thumbnail(tri_id):
        if tri_id < 0:
            thumb_actor.GetProperty().SetOpacity(0.0)
            return
        vtk_img = generate_thumbnail_image(active_cond()['recnum'], tri_id)
        if vtk_img is None:
            thumb_actor.GetProperty().SetOpacity(0.0)
            return
        win_w, win_h = renderWindow.GetSize()
        thumb_w, thumb_h = max(200, int(win_w * THUMB_FRAC_W)), max(260, int(win_h * THUMB_FRAC_H))
        resizer = vtk.vtkImageResize()
        resizer.SetInputData(vtk_img)
        resizer.SetOutputDimensions(thumb_w, thumb_h, 1)
        resizer.Update()
        thumb_mapper.SetInputData(resizer.GetOutput())
        thumb_actor.SetPosition(win_w - thumb_w - MARGIN, MARGIN)
        thumb_actor.GetProperty().SetOpacity(1.0)
        state['last_thumb_tri'] = tri_id

    def refresh_corner_text():
        if state['number_buffer'] is not None:
            return
        cond = active_cond()
        tri = state['current_tri']
        if 0 <= tri < cond['num_triangles']:
            cluster_id = cond['cluster_array'].GetValue(tri)
            a1 = cond['apd1_list'][tri]
            a2 = cond['apd2_list'][tri]
        else:
            cluster_id, a1, a2 = -1, None, None
        cluster_label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        msg = (f"Triangle: {tri}\n"
               f"{cluster_label}\n"
               f"APD1: {f'{a1:.1f} ms' if a1 else '--'}\n"
               f"APD2: {f'{a2:.1f} ms' if a2 else '--'}")
        corner_text.SetInput(msg)

    def update_sphere(new_pos):
        state['sphere_pos'] = new_pos
        sphereSource.SetCenter(new_pos[0], new_pos[1], new_pos[2])
        sphereSource.Update()

        cond = active_cond()
        closest_point = [0.0, 0.0, 0.0]
        cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        cond['cell_locator'].FindClosestPoint(new_pos.tolist(), closest_point, cell_id, sub_id, dist2)
        tri_id = int(cell_id.get())
        state['current_tri'] = tri_id

        refresh_corner_text()
        if len(state['keys_held']) > 0:
            state['needs_update'] = True
        elif tri_id != state['last_thumb_tri']:
            refresh_thumbnail(tri_id)
        renderWindow.Render()

    def compute_move_direction():
        keys = state['keys_held']
        if not keys:
            return np.array([0.0, 0.0, 0.0])
        cam = renderer.GetActiveCamera()
        forward = np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition())
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array(cam.GetViewUp()))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        direction = np.zeros(3)
        if 'Left' in keys: direction -= right
        if 'Right' in keys: direction += right
        if 'Up' in keys: direction += up
        if 'Down' in keys: direction -= up
        n = np.linalg.norm(direction)
        return direction / n if n > 0 else direction

    def switch_condition(new_key):
        if state['active'] == new_key:
            return
        old_cond = conditions[state['active']]
        old_cond['actor_APD'].SetVisibility(0)
        old_cond['actor_wf'].SetVisibility(0)

        state['active'] = new_key
        new_cond = conditions[new_key]
        new_cond['actor_APD'].SetVisibility(1)
        new_cond['actor_wf'].SetVisibility(1)

        title_text.SetInput(new_cond['label'])
        view_text.SetInput(
            f"View: {'BASELINE' if new_key == 'B' else 'LQT'}  [B = Baseline | L = LQT]"
        )

        # Re-locate the blue dot / current triangle against the newly active mesh,
        # keeping its world-space position (same physical spot on the heart).
        update_sphere(state['sphere_pos'])

    def on_key_press(obj, event):
        key = obj.GetKeySym()

        if state['number_buffer'] is not None:
            if key.isdigit():
                state['number_buffer'] += key
                corner_text.SetInput(f"Find triangle: {state['number_buffer']}\nPress Enter to jump")
            elif key == 'Return':
                buf = state['number_buffer']
                state['number_buffer'] = None
                cond = active_cond()
                if buf and 0 <= int(buf) < cond['num_triangles']:
                    target_tri = int(buf)
                    target_pos = cond['triangle_centroids'][target_tri].copy()
                    update_sphere(target_pos)
            renderWindow.Render()
            return

        if key in ('plus', 'equal'):
            renderer.GetActiveCamera().Zoom(1.1)
            renderWindow.Render()
            return
        elif key in ('minus', 'kp_minus'):
            renderer.GetActiveCamera().Zoom(0.9)
            renderWindow.Render()
            return

        if key in ('Left', 'Right', 'Up', 'Down'):
            state['keys_held'].add(key)
        elif key in ('b', 'B'):
            switch_condition('B')
        elif key in ('l', 'L'):
            switch_condition('L')
        elif key in ('f', 'F'):
            state['number_buffer'] = ''
            corner_text.SetInput("Find triangle: \nPress Enter to jump")
            renderWindow.Render()
        elif key == 'Return':
            tri = state['current_tri']
            if tri >= 0:
                update_sphere(active_cond()['triangle_centroids'][tri])
                print(f"Snapped to center of triangle {tri}")

    def on_key_release(obj, event):
        key = obj.GetKeySym()
        if state['number_buffer'] is None and key in ('Left', 'Right', 'Up', 'Down'):
            state['keys_held'].discard(key)
            if len(state['keys_held']) == 0 and state['needs_update']:
                refresh_thumbnail(state['current_tri'])
                state['needs_update'] = False
                renderWindow.Render()

    def on_timer(obj, event):
        if state['number_buffer'] is not None:
            return
        direction = compute_move_direction()
        if np.linalg.norm(direction) > 0:
            new_pos = state['sphere_pos'] + direction * 0.3
            cond = active_cond()
            closest_point = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            cond['cell_locator'].FindClosestPoint(new_pos.tolist(), closest_point, cell_id, sub_id, dist2)
            update_sphere(np.array(closest_point))

    def on_window_resize(obj, event):
        refresh_thumbnail(state['current_tri'])

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetNumberOfLayers(2)
    renderWindow.AddRenderer(renderer)
    renderWindow.AddRenderer(overlay_renderer)
    renderWindow.SetSize(900, 900)
    renderWindow.AddObserver('ModifiedEvent', on_window_resize)

    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    renderer.AddLight(light)

    center_of_mass = conditions['B']['center_of_mass']
    camera = vtk.vtkCamera()
    camera.SetPosition(-80 + center_of_mass[0], center_of_mass[1], center_of_mass[2])
    camera.SetFocalPoint(*center_of_mass)
    renderer.SetActiveCamera(camera)
    renderer.GetActiveCamera().Roll(75)
    renderer.GetActiveCamera().Elevation(60)
    renderer.GetActiveCamera().Roll(20)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.RemoveObservers('CharEvent')

    interactor.AddObserver('KeyPressEvent',   on_key_press)
    interactor.AddObserver('KeyReleaseEvent', on_key_release)
    interactor.AddObserver('TimerEvent',      on_timer)

    refresh_corner_text()
    renderWindow.Render()
    refresh_thumbnail(0)

    interactor.Initialize()
    interactor.CreateRepeatingTimer(25)
    interactor.Start()


if __name__ == '__main__':
    run_wave_explorer()