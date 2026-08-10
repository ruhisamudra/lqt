import vtk
import vtk.util.numpy_support
import sys
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

# ── ON-THE-FLY PLOT → vtkImageData ──────────────────────────────────────────

def generate_thumbnail_image(version, recnum, filenum):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    folder = f"data/{date_subfolder}/{recnum}/timeseries ratiometric raw/"
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

    if filenum < 0 or filenum >= len(files):
        return None

    try:
        if version == "original":
            all_traces = np.array([np.load(os.path.join(folder, f)) for f in files])
            raw_trace = all_traces[filenum]
            raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())
            n_points = len(raw_norm)
            time_ms = np.linspace(0, 1000, n_points)

            raw_smooth = savgol_filter(raw_norm, window_length=11, polyorder=3)
            first_deriv = savgol_filter(np.gradient(raw_smooth), window_length=11, polyorder=3)
            second_deriv = savgol_filter(np.gradient(first_deriv), window_length=11, polyorder=3)

            samples_per_ms = n_points / 1000
            min_distance_samples = int(100 * samples_per_ms)
            upstroke_indices, _ = find_peaks(
                first_deriv, height=0.05, distance=min_distance_samples, prominence=0.05
            )

            steep_indices = []
            repol_indices = []
            for i, up_idx in enumerate(upstroke_indices):
                window_start = up_idx + 5
                window_end = upstroke_indices[i+1] if i+1 < len(upstroke_indices) else len(raw_smooth)
                steep = window_start + np.argmin(first_deriv[window_start:window_end])
                steep_indices.append(steep)
                search_end = min(steep + 30, window_end)
                repol = steep + np.argmax(np.abs(second_deriv[steep:search_end]))
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
            baseline_smooth = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
            corrected_trace = raw_norm - baseline_smooth

            fig, axes = plt.subplots(4, 1, figsize=(4, 6), sharex=False)
            fig.patch.set_facecolor('white')
            axes[0].plot(time_ms, raw_norm, color="black", linewidth=0.8)
            for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
                axes[0].axvline(time_ms[up], color="red", linewidth=1.0)
                axes[0].axvline(time_ms[st], color="magenta", linewidth=1.0)
                axes[0].axvline(time_ms[re], color="blue", linewidth=1.0)
            axes[0].set_ylabel("Signal", fontsize=6)
            axes[0].tick_params(labelsize=5)
            axes[1].plot(time_ms, first_deriv, color="black", linewidth=0.8)
            axes[1].plot(time_ms, second_deriv, color="gray", linewidth=0.6)
            axes[1].axhline(0, color="gray", linewidth=0.5)
            for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
                axes[1].axvline(time_ms[up], color="red", linewidth=1.0)
                axes[1].axvline(time_ms[st], color="magenta", linewidth=1.0)
                axes[1].axvline(time_ms[re], color="blue", linewidth=1.0)
            axes[1].set_ylabel("dF/dt", fontsize=6)
            axes[1].tick_params(labelsize=5)
            axes[2].plot(time_ms, raw_norm, color="black", linewidth=0.8)
            axes[2].plot(time_ms, baseline_smooth, color="green", linewidth=0.8)
            axes[2].set_ylabel("Baseline", fontsize=6)
            axes[2].tick_params(labelsize=5)
            axes[3].plot(time_ms, corrected_trace, color="black", linewidth=0.8)
            axes[3].set_ylabel("Corrected", fontsize=6)
            axes[3].set_xlabel("Time [ms]", fontsize=6)
            axes[3].tick_params(labelsize=5)

        else:  # new pipeline
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
            fft_freqs_arr = fftfreq(n_points, d=1.0/sampling_rate)
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
                b, a = butter(order, cutoff/nyquist, btype='low', analog=False)
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
                next_upstroke = upstroke_indices[i+1] if i+1 < len(upstroke_indices) else n_points - 1
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

# ── SHARED APD COMPUTATION ────────────────────────────────────────────────────

def compute_apd(version, recnum, triangle_index):
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    if version == 'original':
        folder = f"data/{date_subfolder}/{recnum}/timeseries detrended/"
        prefix = "trace_detrended_triangle_"
    else:
        folder = f"data/{date_subfolder}/{recnum}/timeseries corrected new/"
        prefix = "trace_corrected_triangle_"

    filepath = os.path.join(folder, f"{prefix}{triangle_index:04}.npy")
    if not os.path.exists(filepath):
        return None, None

    trace = np.load(filepath)
    trace_SG = savgol_filter(trace, window_length=21, polyorder=9)
    trace_mn = np.convolve(trace, np.ones(5)/5, mode='same')

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

# ── MAIN CORE ENGINE ──────────────────────────────────────────────────────────

def run_wave_explorer(recnum='rec08'):
    t = 0
    show_wireframe = True
    view4c = True

    THUMB_FRAC_W = 0.3
    THUMB_FRAC_H = 0.7
    MARGIN = 16

    mesh_paths = {
        "rec02": "data/2025-07-23/rec02/mesh/",
        "rec07": "data/2025-07-23/rec07/mesh/",
        "rec03": "data/2025-10-22/rec03/mesh/",
        "rec08": "data/2025-10-22/rec08/mesh/"
    }
    mesh_path = mesh_paths.get(recnum, f"data/2025-10-22/{recnum}/mesh/")

    print(f'=== Starting Integrated Cardiac Explorer: {recnum.upper()} ===')
    print('Controls: [Space] = Play/Pause Wave (Hides Plots) | [O] = Original Map | [N] = New Map | [F] = Search Triangle')
    print('Loading 3D mesh architecture...')

    filename = mesh_path + "mesh" + str(t) + ".gltf"
    reader = vtk.vtkGLTFImporter()
    reader.SetFileName(filename)
    reader.Update()

    def extract_polydata(gltf_reader):
        r = gltf_reader.GetRenderer()
        actors = r.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        if isinstance(actor, vtk.vtkActor):
            mapper = actor.GetMapper()
            data = mapper.GetInput()
            if isinstance(data, vtk.vtkPolyData):
                return data
        return None

    polydata = extract_polydata(reader)
    if polydata is None:
        print("ERROR: Could not load valid polydata from GLTF file.")
        return

    num_points = polydata.GetNumberOfPoints()
    center_of_mass = [0.0, 0.0, 0.0]
    for i in range(num_points):
        pt = [0.0, 0.0, 0.0]
        polydata.GetPoint(i, pt)
        center_of_mass[0] += pt[0]
        center_of_mass[1] += pt[1]
        center_of_mass[2] += pt[2]
    center_of_mass = [c / num_points for c in center_of_mass]

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

    print("Preloading and normalizing raw time-series trace signals...")
    date_subfolder = "2025-07-23" if recnum in ["rec02", "rec07"] else "2025-10-22"
    folder_raw = f"data/{date_subfolder}/{recnum}/timeseries ratiometric raw/"
    raw_files = sorted([f for f in os.listdir(folder_raw) if f.endswith(".npy")])
    
    sample_trace = np.load(os.path.join(folder_raw, raw_files[0]))
    n_time_points = len(sample_trace)
    
    signal_matrix = np.zeros((num_triangles, n_time_points))
    for i in range(num_triangles):
        if i < len(raw_files):
            trace = np.load(os.path.join(folder_raw, raw_files[i]))
            trace_smooth = savgol_filter(trace, window_length=5, polyorder=2)
            t_min, t_max = trace_smooth.min(), trace_smooth.max()
            if (t_max - t_min) > 0:
                signal_matrix[i, :] = 1.0 - ((trace_smooth - t_min) / (t_max - t_min))

    wave_intensity_array = vtk.vtkDoubleArray()
    wave_intensity_array.SetName("WaveIntensity")
    wave_intensity_array.SetNumberOfTuples(num_triangles)

    bw_color_func = vtk.vtkColorTransferFunction()
    bw_color_func.AddRGBPoint(0.0, 1.0, 1.0, 1.0)
    bw_color_func.AddRGBPoint(1.0, 0.0, 0.0, 0.0)

    apd_data = {}
    for version in ('original', 'new'):
        print(f"Precomputing APD values ({version.upper()} pipeline)...")
        triangle_APD = vtk.vtkDoubleArray()
        triangle_APD.SetNumberOfTuples(num_triangles)
        apd1_list, apd2_list = [], []

        for i in range(num_triangles):
            a1, a2 = compute_apd(version, recnum, i)
            apd1_list.append(a1)
            apd2_list.append(a2)
            triangle_APD.SetValue(i, (a1 + a2) / 2.0 if (a1 is not None and a2 is not None) else 0.0)

        apd_data[version] = {'array': triangle_APD, 'apd1': apd1_list, 'apd2': apd2_list}

    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(320.0, 0.5, 0.0, 0.0)
    color_func.AddRGBPoint(300.0, 1.0, 0.0, 0.0)
    color_func.AddRGBPoint(250.0, 1.0, 0.5, 0.0)
    color_func.AddRGBPoint(170.0, 1.0, 1.0, 0.0)
    color_func.AddRGBPoint(150.0, 0.0, 1.0, 0.0)
    color_func.AddRGBPoint(120.0, 0.0, 1.0, 0.0)
    color_func.AddRGBPoint(  1.0, 1.0, 1.0, 1.0)
    color_func.AddRGBPoint(  0.0, 1.0, 1.0, 1.0)

    state = {
        'version':       'new',
        'sphere_pos':    triangle_centroids[0].copy(),
        'current_tri':   0,
        'keys_held':     set(),
        'number_buffer': None,
        'last_thumb_tri': -1,
        'needs_update':  False,
        'animating':     False,
        'anim_frame':    0,
        'max_frames':    n_time_points,
    }

    triangle_filter.GetOutput().GetCellData().SetScalars(apd_data[state['version']]['array'])

    renderer = vtk.vtkRenderer()
    renderer.SetLayer(0)
    renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    renderer.SetBackground(1.0, 1.0, 1.0)

    overlay_renderer = vtk.vtkRenderer()
    overlay_renderer.SetLayer(1)
    overlay_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    overlay_renderer.InteractiveOff()
    overlay_renderer.SetBackgroundAlpha(0.0)

    mapper_APD = vtk.vtkPolyDataMapper()
    mapper_APD.SetInputData(triangle_filter.GetOutput())
    mapper_APD.SetScalarRange(0.0, 320.0)
    mapper_APD.SetLookupTable(color_func)
    actor_APD = vtk.vtkActor()
    actor_APD.SetMapper(mapper_APD)
    renderer.AddActor(actor_APD)

    if show_wireframe:
        mapper_wf = vtk.vtkPolyDataMapper()
        mapper_wf.SetInputData(polydata)
        actor_wf = vtk.vtkActor()
        actor_wf.SetMapper(mapper_wf)
        actor_wf.GetProperty().SetRepresentationToWireframe()
        actor_wf.GetProperty().SetColor(0.85, 0.85, 0.85)
        actor_wf.GetProperty().SetLineWidth(1.0)
        renderer.AddActor(actor_wf)

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

    pipeline_text = vtk.vtkTextActor()
    pipeline_text.GetTextProperty().SetFontSize(25)
    pipeline_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    pipeline_text.GetTextProperty().SetFontFamilyToTimes()
    pipeline_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    pipeline_text.SetPosition(0.02, 0.92)
    pipeline_text.SetInput("Pipeline: New  [O = Original | N = New]")
    renderer.AddActor2D(pipeline_text)

    title_text = vtk.vtkTextActor()
    title_text.GetTextProperty().SetFontSize(30)
    title_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    title_text.GetTextProperty().SetFontFamilyToTimes()
    title_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    title_text.SetPosition(0.02, 0.94)
    
    base_titles = {"rec02": "BASELINE (rec02, 07/23/2025)", "rec07": "LQT (rec07, 07/23/2025)", "rec03": "BASELINE (rec03, 10/22/2025)", "rec08": "LQT (rec08, 10/22/2025)"}
    title_text.SetInput(base_titles.get(recnum, f"Dataset: {recnum.upper()}"))
    renderer.AddActor2D(title_text)

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

    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(triangle_filter.GetOutput())
    cell_locator.BuildLocator()

    def refresh_thumbnail(tri_id):
        # Explicitly hide plot textures if we are running the wave loop animation
        if tri_id < 0 or state['animating']:
            thumb_actor.GetProperty().SetOpacity(0.0)
            return
        vtk_img = generate_thumbnail_image(state['version'], recnum, tri_id)
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
        if state['number_buffer'] is not None: return
        tri = state['current_tri']
        if state['animating']:
            ms = int((state['anim_frame'] / state['max_frames']) * 1000)
            corner_text.SetInput(f"Wave Loop playing...\nTimestamp: {ms} ms\nTriangle under focus: {tri}")
            return
        v = state['version']
        a1, a2 = apd_data[v]['apd1'][tri], apd_data[v]['apd2'][tri]
        msg = f"Triangle: {tri}\nAPD1: {f'{a1:.1f} ms' if a1 else '--'}\nAPD2: {f'{a2:.1f} ms' if a2 else '--'}"
        corner_text.SetInput(msg)

    def update_sphere(new_pos):
        state['sphere_pos'] = new_pos
        sphereSource.SetCenter(new_pos[0], new_pos[1], new_pos[2])
        sphereSource.Update()
        
        closest_point = [0.0, 0.0, 0.0]
        cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        cell_locator.FindClosestPoint(new_pos.tolist(), closest_point, cell_id, sub_id, dist2)
        tri_id = int(cell_id.get())
        state['current_tri'] = tri_id
        
        refresh_corner_text()
        if len(state['keys_held']) > 0:
            state['needs_update'] = True
        elif not state['animating'] and tri_id != state['last_thumb_tri']:
            refresh_thumbnail(tri_id)
        renderWindow.Render()

    def compute_move_direction():
        keys = state['keys_held']
        if not keys: return np.array([0.0, 0.0, 0.0])
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

    def on_key_press(obj, event):
        key = obj.GetKeySym()

        if state['number_buffer'] is not None:
            if key.isdigit():
                state['number_buffer'] += key
                corner_text.SetInput(f"Find triangle: {state['number_buffer']}\nPress Enter to jump")
            elif key == 'Return':
                buf = state['number_buffer']
                state['number_buffer'] = None
                if buf and 0 <= int(buf) < num_triangles:
                    target_tri = int(buf)
                    target_pos = triangle_centroids[target_tri].copy()
                    update_sphere(target_pos)
            renderWindow.Render()
            return
        
        if key in ('plus', 'equal'):
            cam = renderer.GetActiveCamera()
            cam.Zoom(1.1)  # Zoom in by 10%
            renderWindow.Render()
            return
        elif key in ('minus', 'kp_minus'):
            cam = renderer.GetActiveCamera()
            cam.Zoom(0.9)  # Zoom out by 10%
            renderWindow.Render()
            return

        if key in ('Left', 'Right', 'Up', 'Down'):
            state['keys_held'].add(key)
        elif key == 'space':
            state['animating'] = not state['animating']
            if state['animating']:
                # Instantly turn off/hide plots panel to maximize full field of view
                thumb_actor.GetProperty().SetOpacity(0.0)
                triangle_filter.GetOutput().GetCellData().SetScalars(wave_intensity_array)
                mapper_APD.SetLookupTable(bw_color_func)
                mapper_APD.SetScalarRange(0.0, 1.0)
            else:
                # Instantly restore static APD map profiles and bring back the thumbnail graph
                triangle_filter.GetOutput().GetCellData().SetScalars(apd_data[state['version']]['array'])
                mapper_APD.SetLookupTable(color_func)
                mapper_APD.SetScalarRange(0.0, 320.0)
                title_text.SetInput(base_titles.get(recnum, f"Dataset: {recnum.upper()}"))
                refresh_thumbnail(state['current_tri'])
            
            triangle_filter.GetOutput().GetCellData().Modified()
            refresh_corner_text()
            renderWindow.Render()
            
        elif key in ('o', 'O'):
            if state['animating']: return
            state['version'] = 'original'
            triangle_filter.GetOutput().GetCellData().SetScalars(apd_data['original']['array'])
            pipeline_text.SetInput("Pipeline: Original  [O = Original | N = New]")
            refresh_thumbnail(state['current_tri'])
            renderWindow.Render()
        elif key in ('n', 'N'):
            if state['animating']: return
            state['version'] = 'new'
            triangle_filter.GetOutput().GetCellData().SetScalars(apd_data['new']['array'])
            pipeline_text.SetInput("Pipeline: New  [O = Original | N = New]")
            refresh_thumbnail(state['current_tri'])
            renderWindow.Render()
        elif key in ('f', 'F'):
            state['number_buffer'] = ''
            corner_text.SetInput("Find triangle: \nPress Enter to jump")
            renderWindow.Render()
        elif key == 'Return':
            tri = state['current_tri']
            if tri >= 0:
                update_sphere(triangle_centroids[tri])
                print(f"Opening plots for triangle {tri} ...")

    def on_key_release(obj, event):
        key = obj.GetKeySym()
        if state['number_buffer'] is None and key in ('Left', 'Right', 'Up', 'Down'):
            state['keys_held'].discard(key)
            if len(state['keys_held']) == 0 and state['needs_update'] and not state['animating']:
                refresh_thumbnail(state['current_tri'])
                renderWindow.Render()

    def on_timer(obj, event):
        if state['number_buffer'] is not None: return
        
        direction = compute_move_direction()
        if np.linalg.norm(direction) > 0:
            new_pos = state['sphere_pos'] + direction * 0.3
            closest_point = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            cell_locator.FindClosestPoint(new_pos.tolist(), closest_point, cell_id, sub_id, dist2)
            update_sphere(np.array(closest_point))

        if state['animating']:
            frame = state['anim_frame']
            for i in range(num_triangles):
                wave_intensity_array.SetValue(i, signal_matrix[i, frame])
            
            wave_intensity_array.Modified()
            triangle_filter.GetOutput().GetCellData().Modified()
            
            ms_timestamp = int((frame / state['max_frames']) * 1000)
            title_text.SetInput(f"{base_titles.get(recnum, recnum.upper())} | Wave: {ms_timestamp} ms")
            refresh_corner_text()
            
            state['anim_frame'] = (frame + 1) % state['max_frames']
            renderWindow.Render()

    def on_window_resize(obj, event):
        if not state['animating']:
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

    if view4c:
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

    renderWindow.Render()
    refresh_thumbnail(0)
    
    interactor.Initialize()
    interactor.CreateRepeatingTimer(25)
    interactor.Start()

if __name__ == '__main__':
    run_wave_explorer(recnum="rec07")