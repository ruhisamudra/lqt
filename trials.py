# # 3D MAP BASED ON APD CALCULATION

# import vtk
# import sys
# import numpy as np
# import os
# import subprocess
# import tempfile
# import atexit
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
# from scipy.fft import fft, fftfreq


# # ── PLOTS FUNCTION (ratiometric raw + derivative, opens on Enter) ─────────────

# def run_apd_plots(version, recnum, filenum):
#     folder = f"data/2025-07-23/{recnum}/timeseries ratiometric raw/"
#     files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

#     if filenum >= len(files):
#         print(f"No raw file for triangle {filenum}")
#         return

#     rec_label = "Normal" if recnum == "rec02" else "LQT"
#     version_label = "Old Pipeline" if version == "old" else "New Pipeline"

#     if version == "old":
#         all_traces = np.array([np.load(os.path.join(folder, f)) for f in files])
#         raw_trace = all_traces[filenum]
#         raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())
#         n_points = len(raw_norm)
#         time_ms = np.linspace(0, 1000, n_points)

#         raw_smooth = savgol_filter(raw_norm, window_length=11, polyorder=3)
#         first_deriv = savgol_filter(np.gradient(raw_smooth), window_length=11, polyorder=3)
#         second_deriv = savgol_filter(np.gradient(first_deriv), window_length=11, polyorder=3)

#         samples_per_ms = n_points / 1000
#         min_distance_samples = int(100 * samples_per_ms)
#         upstroke_indices, _ = find_peaks(
#             first_deriv, height=0.05, distance=min_distance_samples, prominence=0.05
#         )

#         steep_indices = []
#         repol_indices = []
#         for i, up_idx in enumerate(upstroke_indices):
#             window_start = up_idx + 5
#             window_end = upstroke_indices[i+1] if i+1 < len(upstroke_indices) else len(raw_smooth)
#             steep = window_start + np.argmin(first_deriv[window_start:window_end])
#             steep_indices.append(steep)
#             search_end = min(steep + 30, window_end)
#             repol = steep + np.argmax(np.abs(second_deriv[steep:search_end]))
#             repol_indices.append(repol)

#         baseline_trace = raw_norm.copy()
#         for up, re in zip(upstroke_indices, repol_indices):
#             baseline_trace[up:re] = np.nan
#         x_all = np.arange(n_points)
#         x_valid = x_all[~np.isnan(baseline_trace)]
#         y_valid = baseline_trace[~np.isnan(baseline_trace)]
#         baseline_interpolated = np.interp(x_all, x_valid, y_valid)
#         baseline_smooth = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
#         corrected_trace = raw_norm - baseline_smooth

#         fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
#         axes[0].plot(time_ms, raw_norm, color="black", linewidth=1.5)
#         for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
#             axes[0].axvline(time_ms[up], color="red", linewidth=1.5)
#             axes[0].axvline(time_ms[st], color="magenta", linewidth=1.5)
#             axes[0].axvline(time_ms[re], color="blue", linewidth=1.5)
#         axes[0].set_xlabel("Time [ms]")
#         axes[0].set_ylabel("Normalized Signal")

#         axes[1].plot(time_ms, first_deriv, color="black", linewidth=1.5)
#         axes[1].plot(time_ms, second_deriv, color="black", linewidth=1.5)
#         axes[1].axhline(0, color="gray", linewidth=1)
#         for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
#             axes[1].axvline(time_ms[up], color="red", linewidth=1.5)
#             axes[1].axvline(time_ms[st], color="magenta", linewidth=1.5)
#             axes[1].axvline(time_ms[re], color="blue", linewidth=1.5)
#         axes[1].set_xlabel("Time [ms]")
#         axes[1].set_ylabel("dF/dt")

#         axes[2].plot(time_ms, raw_norm, color="black", linewidth=1.5, label="Raw Trace")
#         axes[2].plot(time_ms, baseline_smooth, color="green", linewidth=1.5, label="Fitted Baseline")
#         axes[2].set_xlabel("Time [ms]")
#         axes[2].set_ylabel("Normalized Signal")
#         axes[2].legend(loc="upper right", fontsize=8)

#         axes[3].plot(time_ms, corrected_trace, color="black", linewidth=1.5)
#         axes[3].set_xlabel("Time [ms]")
#         axes[3].set_ylabel("Corrected Signal")

#         APD = time_ms[repol_indices] - time_ms[upstroke_indices]
#         print("APD per beat (ms):", APD)

#     elif version == "new":
#         raw_trace = np.load(os.path.join(folder, files[filenum]))
#         n_points = len(raw_trace)
#         time_ms = np.linspace(0, 1000, n_points)
#         samples_per_ms = n_points / 1000.0
#         sampling_rate = 250.0
#         MIN_APD_MS = 140.0
#         min_apd_frames = int(MIN_APD_MS * samples_per_ms)
#         MAX_PHASE_WINDOW_MS = 500.0
#         max_phase_frames = int(MAX_PHASE_WINDOW_MS * samples_per_ms)

#         raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

#         fft_vals = np.abs(fft(raw_norm))
#         fft_freqs_arr = fftfreq(n_points, d=1.0/sampling_rate)
#         half_points = n_points // 2
#         freqs_side = fft_freqs_arr[:half_points]
#         vals_side = fft_vals[:half_points]
#         fft_slope = np.gradient(vals_side)
#         search_start_idx = np.searchsorted(freqs_side, 3.0)
#         search_end_idx = np.searchsorted(freqs_side, 50.0)
#         optimal_idx = search_start_idx + np.argmax(fft_slope[search_start_idx:search_end_idx])
#         CUTOFF_FREQUENCY = float(np.clip(freqs_side[optimal_idx], 15.0, 45.0))
#         print(f"--> Automated FFT Analysis complete. Optimal Cutoff: {CUTOFF_FREQUENCY:.2f} Hz")

#         def butter_lowpass_filter(data, cutoff, fs, order=4):
#             nyquist = 0.5 * fs
#             b, a = butter(order, cutoff/nyquist, btype='low', analog=False)
#             return filtfilt(b, a, data)

#         raw_smooth_butter = butter_lowpass_filter(raw_norm, CUTOFF_FREQUENCY, sampling_rate)
#         first_deriv_butter = np.gradient(raw_smooth_butter)
#         second_deriv_butter = np.gradient(first_deriv_butter)

#         min_distance_samples = int(100 * samples_per_ms)
#         upstroke_indices, _ = find_peaks(
#             first_deriv_butter, height=0.05,
#             distance=min_distance_samples, prominence=0.05
#         )

#         steep_indices = []
#         repol_indices = []
#         for i, up_idx in enumerate(upstroke_indices):
#             window_start = up_idx + min_apd_frames
#             next_upstroke = upstroke_indices[i+1] if i+1 < len(upstroke_indices) else n_points - 1
#             window_end = min(up_idx + max_phase_frames, next_upstroke, n_points - 1)
#             if window_start >= window_end:
#                 window_start = up_idx + 5
#             steep = window_start + np.argmin(first_deriv_butter[window_start:window_end])
#             steep_indices.append(steep)
#             search_end = min(steep + 25, window_end)
#             repol = steep + (np.argmax(second_deriv_butter[steep:search_end]) if search_end > steep else 0)
#             repol_indices.append(repol)

#         baseline_trace = raw_norm.copy()
#         for up, re in zip(upstroke_indices, repol_indices):
#             baseline_trace[up:re] = np.nan
#         x_all = np.arange(n_points)
#         x_valid = x_all[~np.isnan(baseline_trace)]
#         y_valid = baseline_trace[~np.isnan(baseline_trace)]
#         baseline_interpolated = np.interp(x_all, x_valid, y_valid)
#         baseline_smooth_savgol = savgol_filter(baseline_interpolated, window_length=21, polyorder=3)
#         corrected_trace_savgol = raw_norm - baseline_smooth_savgol

#         APD = time_ms[repol_indices] - time_ms[upstroke_indices]
#         print("Bounded APD per beat (ms):", APD)

#         fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
#         axes[0].plot(time_ms, raw_norm, color="lightgray", linewidth=1, label="Raw Noise")
#         axes[0].plot(time_ms, raw_smooth_butter, color="black", linewidth=1.5, label="Butterworth Filtered")
#         for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
#             axes[0].axvline(time_ms[up], color="red", linewidth=1.5)
#             axes[0].axvline(time_ms[st], color="magenta", linewidth=1.5)
#             axes[0].axvline(time_ms[re], color="blue", linewidth=1.5)
#         axes[0].set_xlabel("Time [ms]")
#         axes[0].set_ylabel("Normalized Signal")
#         axes[0].legend(loc="upper right")

#         axes[1].plot(time_ms, first_deriv_butter, color="black", linewidth=1.5)
#         axes[1].plot(time_ms, second_deriv_butter, color="black", linewidth=1.5)
#         axes[1].axhline(0, color="gray", linewidth=1)
#         for up, st, re in zip(upstroke_indices, steep_indices, repol_indices):
#             axes[1].axvline(time_ms[up], color="red", linewidth=1.5)
#             axes[1].axvline(time_ms[st], color="magenta", linewidth=1.5)
#             axes[1].axvline(time_ms[re], color="blue", linewidth=1.5)
#         axes[1].set_xlabel("Time [ms]")
#         axes[1].set_ylabel("dF/dt")

#         axes[2].plot(time_ms, raw_norm, color="black", linewidth=1.5, label="Raw Trace")
#         axes[2].plot(time_ms, baseline_smooth_savgol, color="green", linewidth=1.5, label="Fitted Baseline (SavGol)")
#         axes[2].set_xlabel("Time [ms]")
#         axes[2].set_ylabel("Normalized Signal")
#         axes[2].legend(loc="upper right", fontsize=8)

#         axes[3].plot(time_ms, corrected_trace_savgol, color="black", linewidth=1.5)
#         axes[3].set_xlabel("Time [ms]")
#         axes[3].set_ylabel("Corrected Signal")

#         fig2, ax2 = plt.subplots(figsize=(8, 4))
#         ax2.plot(freqs_side, vals_side, color="black", linewidth=1.5, label="FFT Power Spectrum")
#         ax2.axvline(CUTOFF_FREQUENCY, color="red", linestyle="--", linewidth=1.5,
#                     label=f"Auto-Cutoff ({CUTOFF_FREQUENCY:.2f} Hz)")
#         ax2.set_xlabel("Frequency (Hz)")
#         ax2.set_ylabel("Amplitude")
#         ax2.set_title("Frequency Spectrum & Automated Filter Selection")
#         ax2.set_xlim(0, 125)
#         ax2.legend(loc="upper right")
#         fig2.tight_layout()
#         tmp2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
#         fig2.savefig(tmp2.name, dpi=150, bbox_inches='tight')
#         plt.close(fig2)
#         atexit.register(os.unlink, tmp2.name)
#         subprocess.Popen(['open', tmp2.name])

#     fig.suptitle(f"Triangle {filenum} | {rec_label} | {version_label}", fontsize=13, y=1.01)
#     fig.tight_layout()
#     tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
#     fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     atexit.register(os.unlink, tmp.name)
#     subprocess.Popen(['open', tmp.name])
#     print(f"Plot opened for triangle {filenum}")


# # ── SHARED APD COMPUTATION (detrended/corrected + threshold crossing) ─────────

# def compute_apd(version, recnum, triangle_index):
#     if version == 'old':
#         folder = f"data/2025-07-23/{recnum}/timeseries detrended/"
#         prefix = "trace_detrended_triangle_"
#     else:
#         folder = f"data/2025-07-23/{recnum}/timeseries corrected new/"
#         prefix = "trace_corrected_triangle_"

#     filepath = os.path.join(folder, f"{prefix}{triangle_index:04}.npy")
#     if not os.path.exists(filepath):
#         return None, None

#     trace = np.load(filepath)
#     trace_SG = savgol_filter(trace, window_length=21, polyorder=9)
#     trace_mn = np.convolve(trace, np.ones(5)/5, mode='same')

#     baseline = np.mean(trace_SG[60:120])
#     peak = np.max(trace_mn)
#     if peak - baseline == 0:
#         return None, None

#     trace_norm    = (trace    - baseline) / (peak - baseline)
#     trace_SG_norm = (trace_SG - baseline) / (peak - baseline)
#     trace_mn_norm = (trace_mn - baseline) / (peak - baseline)

#     APD_threshold = 0.1
#     up_crossings = np.where(
#         (trace_mn_norm[:-1] < APD_threshold) &
#         (trace_mn_norm[1:] >= APD_threshold) &
#         (np.abs(trace_mn_norm[:-1] - trace_mn_norm[1:]) > 0.05)
#     )[0]
#     down_crossings = np.where(
#         (trace_SG_norm[:-1] >= APD_threshold) &
#         (trace_SG_norm[1:] < APD_threshold)
#     )[0]

#     n_up = len(up_crossings)
#     n_down = len(down_crossings)

#     if(n_up == 2 & n_down == 2):
#         apd1 = (down_crossings[0] - up_crossings[0]) * 4.0
#         apd2 = (down_crossings[1] - up_crossings[1]) * 4.0
#         return apd1, apd2
#     else:
#         return None, None


# # ── MAIN 3D MAP ───────────────────────────────────────────────────────────────

# def run_APD_map(version='old', LQT=False):

#     t = 0
#     show_wireframe = True
#     show_APD = True
#     write_images = True
#     view4c = True
#     view1 = view2 = view3 = view4 = view4b = view5 = False

#     path1 = "data/2025-07-23/rec02/mesh/"
#     path2 = "data/2025-07-23/rec07/mesh/"
#     recnum = "rec07" if LQT else "rec02"
#     mesh_path = path2 if LQT else path1
#     foldername_png = "snapshots/"

#     print(f'Running: version={version}, LQT={LQT}')
#     print('Loading data ...')

#     filename = mesh_path + "mesh" + str(t) + ".gltf"
#     reader1 = vtk.vtkGLTFImporter()
#     reader1.SetFileName(filename)
#     reader1.Update()
#     reader2 = vtk.vtkGLTFImporter()
#     reader2.SetFileName(filename)
#     reader2.Update()

#     def extract_polydata(reader):
#         r = reader.GetRenderer()
#         actors = r.GetActors()
#         actors.InitTraversal()
#         actor = actors.GetNextActor()
#         if isinstance(actor, vtk.vtkActor):
#             mapper = actor.GetMapper()
#             data = mapper.GetInput()
#             if isinstance(data, vtk.vtkPolyData):
#                 return data
#         return None

#     polydata1 = extract_polydata(reader1)
#     polydata2 = extract_polydata(reader2)
#     if polydata1 is None or polydata2 is None:
#         print("ERROR: Could not extract polydata.")
#         return

#     numPoints = polydata2.GetNumberOfPoints()
#     centerOfMass = [0.0, 0.0, 0.0]
#     for i in range(numPoints):
#         pt = [0.0, 0.0, 0.0]
#         polydata2.GetPoint(i, pt)
#         centerOfMass[0] += pt[0]
#         centerOfMass[1] += pt[1]
#         centerOfMass[2] += pt[2]
#     centerOfMass = [c / numPoints for c in centerOfMass]

#     renderer = vtk.vtkRenderer()

#     # ── Triangle filter & APD precomputation ──────────────────────────────────
#     triangle_filter = vtk.vtkTriangleFilter()
#     triangle_filter.SetInputData(polydata1)
#     triangle_filter.Update()
#     num_triangles = triangle_filter.GetOutput().GetNumberOfCells()

#     triangle_APD = vtk.vtkDoubleArray()
#     triangle_APD.SetName("TriangleAPD")
#     triangle_APD.SetNumberOfTuples(num_triangles)

#     apd1_list = []
#     apd2_list = []
#     triangle_centroids = []

#     print("Precomputing APD for all triangles...")
#     for i in range(num_triangles):
#         cell = triangle_filter.GetOutput().GetCell(i)
#         pts = cell.GetPoints()
#         p0 = np.array(pts.GetPoint(0))
#         p1 = np.array(pts.GetPoint(1))
#         p2 = np.array(pts.GetPoint(2))
#         triangle_centroids.append((p0+p1+p2)/3)

#         a1, a2 = compute_apd(version, recnum, i)
#         apd1_list.append(a1)
#         apd2_list.append(a2)

#         if a1 is not None and a2 is not None:
#             triangle_APD.SetValue(i, (a1 + a2) / 2.0)
#         else:
#             triangle_APD.SetValue(i, 0.0)

#     triangle_centroids = np.array(triangle_centroids)

#     colored = sum(1 for i in range(num_triangles) if triangle_APD.GetValue(i) > 0)
#     white   = num_triangles - colored
#     print(f"Colored: {colored}  White: {white}  Total: {num_triangles}  "
#           f"Coverage: {colored/num_triangles*100:.1f}%")

#     triangle_filter.GetOutput().GetCellData().SetScalars(triangle_APD)

#     color_func = vtk.vtkColorTransferFunction()
#     color_func.AddRGBPoint(320.0, 0.5, 0.0, 0.0)
#     color_func.AddRGBPoint(300.0, 1.0, 0.0, 0.0)
#     color_func.AddRGBPoint(250.0, 1.0, 0.5, 0.0)
#     color_func.AddRGBPoint(170.0, 1.0, 1.0, 0.0)
#     color_func.AddRGBPoint(150.0, 0.0, 1.0, 0.0)
#     color_func.AddRGBPoint(120.0, 0.0, 1.0, 0.0)
#     color_func.AddRGBPoint(  1.0, 1.0, 1.0, 1.0)
#     color_func.AddRGBPoint(  0.0, 1.0, 1.0, 1.0)

#     mapper_APD = vtk.vtkPolyDataMapper()
#     mapper_APD.SetInputData(triangle_filter.GetOutput())
#     mapper_APD.SetScalarRange(0.0, 320.0)
#     mapper_APD.SetLookupTable(color_func)
#     actor_APD = vtk.vtkActor()
#     actor_APD.SetMapper(mapper_APD)
#     renderer.AddActor(actor_APD)

#     if show_wireframe:
#         mapper_wf = vtk.vtkPolyDataMapper()
#         mapper_wf.SetInputData(polydata1)
#         actor_wf = vtk.vtkActor()
#         actor_wf.SetMapper(mapper_wf)
#         actor_wf.GetProperty().SetRepresentationToWireframe()
#         actor_wf.GetProperty().SetColor(0.5, 0.5, 0.5)
#         actor_wf.GetProperty().SetLineWidth(1.0)
#         renderer.AddActor(actor_wf)

#     # ── Cell locator ──────────────────────────────────────────────────────────
#     cell_locator = vtk.vtkCellLocator()
#     cell_locator.SetDataSet(triangle_filter.GetOutput())
#     cell_locator.BuildLocator()

#     # ── Sphere ────────────────────────────────────────────────────────────────
#     start_pos = triangle_centroids[0].copy()

#     sphereSource = vtk.vtkSphereSource()
#     sphereSource.SetRadius(0.25)
#     sphereSource.SetPhiResolution(30)
#     sphereSource.SetThetaResolution(60)
#     sphereSource.SetCenter(start_pos[0], start_pos[1], start_pos[2])
#     sphereSource.Update()

#     sphere_mapper = vtk.vtkPolyDataMapper()
#     sphere_mapper.SetInputConnection(sphereSource.GetOutputPort())
#     actor_sphere = vtk.vtkActor()
#     actor_sphere.SetMapper(sphere_mapper)
#     actor_sphere.GetProperty().SetColor(0.0, 0.0, 0.6)
#     actor_sphere.GetProperty().SetAmbient(0.5)
#     renderer.AddActor(actor_sphere)

#     # ── Corner text ───────────────────────────────────────────────────────────
#     corner_text = vtk.vtkTextActor()
#     corner_text.GetTextProperty().SetFontSize(16)
#     corner_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
#     corner_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
#     corner_text.SetPosition(0.02, 0.02)
#     corner_text.SetInput("")
#     renderer.AddActor2D(corner_text)

#     # ── State ─────────────────────────────────────────────────────────────────
#     state = {
#         'sphere_pos': start_pos.copy(),
#         'current_tri': 0,
#         'keys_held': set(),
#         'number_buffer': None,  # None = not in find mode, string = actively typing
#     }

#     STEP_SIZE = 0.3

#     def find_triangle_at(pos):
#         closest_point = [0.0, 0.0, 0.0]
#         cell_id = vtk.reference(0)
#         sub_id = vtk.reference(0)
#         dist2 = vtk.reference(0.0)
#         cell_locator.FindClosestPoint(pos.tolist(), closest_point, cell_id, sub_id, dist2)
#         cp = np.array(closest_point)
#         tri_id = int(cell_id.get())

#         cell = triangle_filter.GetOutput().GetCell(tri_id)
#         pts = cell.GetPoints()
#         p0 = np.array(pts.GetPoint(0))
#         p1 = np.array(pts.GetPoint(1))
#         p2 = np.array(pts.GetPoint(2))

#         v0 = p1 - p0
#         v1 = p2 - p0
#         v2 = cp - p0
#         d00 = np.dot(v0, v0)
#         d01 = np.dot(v0, v1)
#         d11 = np.dot(v1, v1)
#         d20 = np.dot(v2, v0)
#         d21 = np.dot(v2, v1)
#         denom = d00 * d11 - d01 * d01
#         if abs(denom) < 1e-10:
#             return -1, cp
#         v = (d11 * d20 - d01 * d21) / denom
#         w = (d00 * d21 - d01 * d20) / denom
#         u = 1.0 - v - w

#         if any(c < 0.02 for c in [u, v, w]):
#             return -1, cp
#         return tri_id, cp

#     def project_and_move(pos, direction):
#         new_pos = pos + direction * STEP_SIZE
#         closest_point = [0.0, 0.0, 0.0]
#         cell_id = vtk.reference(0)
#         sub_id = vtk.reference(0)
#         dist2 = vtk.reference(0.0)
#         cell_locator.FindClosestPoint(new_pos.tolist(), closest_point, cell_id, sub_id, dist2)
#         return np.array(closest_point)

#     def update_sphere(new_pos):
#         state['sphere_pos'] = new_pos
#         sphereSource.SetCenter(new_pos[0], new_pos[1], new_pos[2])
#         sphereSource.Update()
#         tri_id, _ = find_triangle_at(new_pos)
#         state['current_tri'] = tri_id
#         # Only update corner text if not in find mode
#         if state['number_buffer'] is None:
#             if tri_id < 0:
#                 corner_text.SetInput("")
#             else:
#                 a1 = apd1_list[tri_id]
#                 a2 = apd2_list[tri_id]
#                 if a1 is None or a2 is None:
#                     msg = f"Triangle: {tri_id}\nAPD1: --\nAPD2: --"
#                 else:
#                     msg = f"Triangle: {tri_id}\nAPD1: {a1:.1f} ms\nAPD2: {a2:.1f} ms"
#                 corner_text.SetInput(msg)
#         renderWindow.Render()

#     def compute_move_direction():
#         keys = state['keys_held']
#         if not keys:
#             return np.array([0.0, 0.0, 0.0])
#         cam = renderer.GetActiveCamera()
#         view_up = np.array(cam.GetViewUp())
#         focal = np.array(cam.GetFocalPoint())
#         pos_cam = np.array(cam.GetPosition())
#         forward = focal - pos_cam
#         forward /= np.linalg.norm(forward)
#         right = np.cross(forward, view_up)
#         right /= np.linalg.norm(right)
#         up = np.cross(right, forward)
#         up /= np.linalg.norm(up)
#         direction = np.zeros(3)
#         if 'Left'  in keys: direction -= right
#         if 'Right' in keys: direction += right
#         if 'Up'    in keys: direction += up
#         if 'Down'  in keys: direction -= up
#         n = np.linalg.norm(direction)
#         return direction / n if n > 0 else direction

#     # ── Render window ─────────────────────────────────────────────────────────
#     renderWindow = vtk.vtkRenderWindow()
#     renderWindow.AddRenderer(renderer)
#     renderWindow.SetSize(800, 800)

#     light = vtk.vtkLight()
#     light.SetIntensity(1)
#     light.SetLightTypeToHeadlight()
#     renderer.AddLight(light)
#     renderer.SetBackground(1.0, 1.0, 1.0)

#     camera = vtk.vtkCamera()
#     if view4c:
#         camera.SetPosition(-80 + centerOfMass[0], centerOfMass[1], centerOfMass[2])
#         camera.SetFocalPoint(*centerOfMass)
#         renderer.SetActiveCamera(camera)
#         renderer.GetActiveCamera().Roll(75)
#         renderer.GetActiveCamera().Elevation(60)
#         renderer.GetActiveCamera().Roll(20)

#     interactor = vtk.vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(renderWindow)

#     # Disable VTK default key bindings that interfere with number keys
#     interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
#     interactor.RemoveObservers('CharEvent')

#     def on_key_press(obj, event):
#         key = obj.GetKeySym()

#         # ── Find mode: absorb ALL keypresses, nothing leaks to VTK ───────────
#         if state['number_buffer'] is not None:
#             if key.isdigit():
#                 state['number_buffer'] += key
#                 corner_text.SetInput(f"Find triangle: {state['number_buffer']}\nPress Enter to jump, F to cancel")
#                 renderWindow.Render()
#             elif key == 'Return':
#                 buf = state['number_buffer']
#                 state['number_buffer'] = None
#                 if buf == '':
#                     corner_text.SetInput("")
#                     renderWindow.Render()
#                     return
#                 target = int(buf)
#                 if 0 <= target < num_triangles:
#                     update_sphere(triangle_centroids[target].copy())
#                     print(f"Jumped to triangle {target}")
#                 else:
#                     print(f"Triangle {target} out of range (0-{num_triangles-1})")
#                     corner_text.SetInput(f"Triangle {target} not found\n(max: {num_triangles-1})")
#                     renderWindow.Render()
#             elif key == 'BackSpace':
#                 state['number_buffer'] = state['number_buffer'][:-1]
#                 corner_text.SetInput(f"Find triangle: {state['number_buffer']}\nPress Enter to jump, F to cancel")
#                 renderWindow.Render()
#             elif key == 'f' or key == 'F':
#                 # cancel find mode
#                 state['number_buffer'] = None
#                 corner_text.SetInput("")
#                 renderWindow.Render()
#             # block everything else — no leaking to VTK
#             return

#         # ── Normal mode ───────────────────────────────────────────────────────
#         if key in ('Left', 'Right', 'Up', 'Down'):
#             state['keys_held'].add(key)
#         elif key == 'f' or key == 'F':
#             state['number_buffer'] = ''
#             corner_text.SetInput("Find triangle: \nPress Enter to jump, F to cancel")
#             renderWindow.Render()
#         elif key == 'Return':
#             tri = state['current_tri']
#             if tri >= 0:
#                 snap_pos = triangle_centroids[tri]
#                 update_sphere(snap_pos)
#                 print(f"Opening plots for triangle {tri} ...")
#                 run_apd_plots(version, recnum, tri)

#     def on_key_release(obj, event):
#         key = obj.GetKeySym()
#         # Only release movement keys if not in find mode
#         if state['number_buffer'] is None:
#             if key in ('Left', 'Right', 'Up', 'Down'):
#                 state['keys_held'].discard(key)

#     def on_timer(obj, event):
#         # Don't move sphere while in find mode
#         if state['number_buffer'] is not None:
#             return
#         direction = compute_move_direction()
#         if np.linalg.norm(direction) == 0:
#             return
#         new_pos = project_and_move(state['sphere_pos'], direction)
#         update_sphere(new_pos)

#     interactor.AddObserver('KeyPressEvent', on_key_press)
#     interactor.AddObserver('KeyReleaseEvent', on_key_release)
#     interactor.AddObserver('TimerEvent', on_timer)

#     renderWindow.Render()
#     interactor.Initialize()
#     interactor.CreateRepeatingTimer(30)

#     if write_images:
#         windowToImageFilter = vtk.vtkWindowToImageFilter()
#         windowToImageFilter.SetInput(renderWindow)
#         windowToImageFilter.Update()
#         writer = vtk.vtkPNGWriter()
#         writer.SetFileName(foldername_png + f"{version}_LQT{LQT}_{t:03d}.png")
#         writer.SetInputConnection(windowToImageFilter.GetOutputPort())
#         writer.Write()

#     interactor.Start()

# # ── CALL ──────────────────────────────────────────────────────────────────────
# run_APD_map("new", True)



# # SAVES TIMESERIES DETRENDED NPY FILES FROM RATIOMETRIC RAW

# import numpy as np
# import os
# from scipy.signal import find_peaks, savgol_filter

# # ─── Settings ──────────────────────────────────────────────────────────────

# recnum = "rec08" # change here
# input_folder  = f"data/2025-10-22/{recnum}/timeseries ratiometric raw/"
# output_folder = f"data/2025-10-22/{recnum}/timeseries detrended/"

# os.makedirs(output_folder, exist_ok = True)

# files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
# print(f"Found {len(files)} traces to process...")

# # ─── Core Function (Old Pipeline) ─────────────────────────────────────────

# def detrend_trace(raw_trace):

#     n_points = len(raw_trace)
#     samples_per_ms = n_points / 1000.0

#     # Normalize
#     raw_norm = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min())

#     # Smooth for detection
#     raw_smooth = savgol_filter(raw_norm, window_length = 11, polyorder = 3)

#     # First and second derivatives
#     first_deriv  = savgol_filter(np.gradient(raw_smooth),  window_length = 11, polyorder = 3)
#     second_deriv = savgol_filter(np.gradient(first_deriv), window_length = 11, polyorder = 3)

#     # Detect upstrokes
#     min_distance_samples = int(100 * samples_per_ms)
#     upstroke_indices, _ = find_peaks(
#         first_deriv,
#         height     = 0.05,
#         distance   = min_distance_samples,
#         prominence = 0.05
#     )

#     # Detect steep downslope and repolarization for each beat
#     steep_indices = []
#     repol_indices = []

#     for i, up_idx in enumerate(upstroke_indices):
#         window_start = up_idx + 5

#         if i + 1 < len(upstroke_indices):
#             window_end = upstroke_indices[i + 1]
#         else:
#             window_end = len(raw_smooth)

#         # Safety check: skip this beat if window is empty or invalid
#         if window_start >= window_end:
#             continue

#         steep = window_start + np.argmin(first_deriv[window_start:window_end])
#         steep_indices.append(steep)

#         search_end = min(steep + 30, window_end)

#         # Safety check again before second argmax
#         if search_end <= steep:
#             repol_indices.append(steep)
#             continue

#         repol = steep + np.argmax(np.abs(second_deriv[steep:search_end]))
#         repol_indices.append(repol)

#     # Baseline correction
#     baseline_trace = raw_norm.copy()
#     for up, re in zip(upstroke_indices, repol_indices):
#         baseline_trace[up:re] = np.nan

#     nan_mask = np.isnan(baseline_trace)
#     x_all   = np.arange(n_points)
#     x_valid = x_all[~nan_mask]
#     y_valid = baseline_trace[~nan_mask]

#     if len(x_valid) < 2:
#         return raw_norm  # fallback if correction not possible

#     baseline_interpolated = np.interp(x_all, x_valid, y_valid)
#     baseline_smooth = savgol_filter(baseline_interpolated, window_length = 21, polyorder = 3)
#     corrected_trace = raw_norm - baseline_smooth

#     return corrected_trace

# # ─── Batch Processing ──────────────────────────────────────────────────────

# for idx, filename in enumerate(files):

#     raw_trace = np.load(os.path.join(input_folder, filename))
#     detrended_trace = detrend_trace(raw_trace)

#     output_filename = f"trace_detrended_triangle_{idx:04d}.npy"
#     np.save(os.path.join(output_folder, output_filename), detrended_trace)

#     if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
#         print(f"Progress: {idx + 1} / {len(files)}")

# print("Done! Detrended traces saved to:", output_folder)



# SAVE TIMESERIES CORRECTED NEW NPY FILES FROM RATIOMETRIC RAW

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