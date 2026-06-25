# lqt

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
   signal quality *(work in progress)*

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

## Notes

Analysis compares normal and Long QT recordings across two experimental
sessions. The pipeline uses FFT-based frequency profiling and a Butterworth
low-pass filter for signal smoothing prior to APD feature extraction.
