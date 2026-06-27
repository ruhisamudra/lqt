# LQT Cardiac Optical Mapping Analysis

Analysis pipeline for cardiac action potential duration (APD) detection
in baseline and Long QT (LQT) optical mapping data.

## Pipeline Overview

Run scripts in order:

1. `genfiles.py` — preprocessing pipeline; generates required input files
   from raw ratiometric data (not included, available on request)
2. `01_raw_traces_plots.py` — plots individual time series traces with
   detected upstroke, steepest downslope, and end of repolarization markers
3. `02_show_APD-map.py` — projects APD values onto a 3D heart mesh;
   shows coverage and spatial distribution of action potential durations
4. `03_clustering.py` — morphology-based clustering of traces to classify
   signal quality and common LQT features *(work in progress)*

## Data

Raw ratiometric `.npy` trace files and mesh data are not included in this
repository. Contact [your email] for access.

Two recording sessions:
- **07/23/2025**: rec02 (baseline), rec07 (LQT)
- **10/22/2025**: rec03 (baseline), rec08 (LQT)

## Requirements

- numpy
- matplot
- scipy
- vtk

## Controls & Features

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause the cardiac wave animation (hides plots for full view) |
| `F` | Search for a triangle by number — type the ID and press Enter to jump to it |
| `Enter` | Confirm triangle search / refresh plots for current triangle |
| `O` | Switch to **Original** processing pipeline |
| `N` | Switch to **New** processing pipeline |
| `↑ ↓ ← →` | Move the selection sphere across the 3D mesh |
| `+` / `=` | Zoom in |
| `-` | Zoom out |

### Features

- **3D Cardiac Mesh Viewer** — Renders the heart mesh with APD (Action Potential Duration) color mapping
- **Wave Animation** — Press Space to play a real-time fluorescence wave propagating across the mesh
- **APD Color Map** — Triangles are colored from white → green → yellow → red based on APD duration (0–320 ms)
- **Triangle Inspector** — Move the blue sphere to any triangle to instantly see its signal trace, derivatives, baseline correction, and corrected trace in a side panel
- **Pipeline Comparison** — Switch between Original and New signal processing pipelines on the fly with O/N keys
- **Triangle Search** — Press F, type a triangle number, and press Enter to jump the sphere directly to that triangle
- **APD Readout** — Corner display shows the current triangle ID, APD1, and APD2 values in milliseconds
