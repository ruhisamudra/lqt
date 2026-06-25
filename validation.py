# Fully Rigorous APD10 Validation Script
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def validate_apd10(version, recnum, filenum):
    # ─── 1. Load Detrended/Corrected Trace ───────────────────────────────────
    if version == "old":
        folder = f"data/2025-07-23/{recnum}/timeseries detrended/"
        prefix = "trace_detrended_triangle_"
    else:
        folder = f"data/2025-07-23/{recnum}/timeseries corrected new/"
        prefix = "trace_corrected_triangle_"

    filepath = os.path.join(folder, f"{prefix}{filenum:04d}.npy")

    if not os.path.exists(filepath):
        print(f"❌ CRITICAL ERROR: No file found at {filepath}")
        return None

    trace = np.load(filepath)
    n_points = len(trace)
    
    # Calculate exact time properties
    time_ms = np.linspace(0, 1000, n_points)  
    calculated_dt = 1000.0 / (n_points - 1) if n_points > 1 else 0

    # ─── 2. Normalize (Identical to compute_apd) ─────────────────────────────
    trace_SG = savgol_filter(trace, window_length=21, polyorder=9)
    trace_mn = np.convolve(trace, np.ones(5)/5, mode='same')

    baseline = np.mean(trace_SG[60:120])
    peak = np.max(trace_mn)

    if peak - baseline == 0:
        print("❌ CRITICAL ERROR: Flat trace detected. Division by zero in normalization.")
        return None

    trace_norm    = (trace    - baseline) / (peak - baseline)
    trace_SG_norm = (trace_SG - baseline) / (peak - baseline)
    trace_mn_norm = (trace_mn - baseline) / (peak - baseline)

    APD_threshold = 0.1

    # ─── 3. Find Crossings (Identical to compute_apd) ──────────────────────────
    up_crossings = np.where(
        (trace_mn_norm[:-1] < APD_threshold) &
        (trace_mn_norm[1:] >= APD_threshold) &
        (np.abs(trace_mn_norm[:-1] - trace_mn_norm[1:]) > 0.05)
    )[0]

    down_crossings = np.where(
        (trace_SG_norm[:-1] >= APD_threshold) &
        (trace_SG_norm[1:] < APD_threshold)
    )[0]

    n_up = len(up_crossings)
    n_down = len(down_crossings)

    # ─── 4. Structural Map Simulation Validation ────────────────────────────
    print("=" * 65)
    print(f"APD10 VALIDATION REPORT | Triangle: {filenum:04d} | Record: {recnum}")
    print(f"Pipeline Version: {version.upper()} | Trace Length: {n_points} frames")
    print("=" * 65)
    print(f"Baseline (Frames 60-120 Avg): {baseline:.4f}")
    print(f"Peak Value (Moving Avg Max):  {peak:.4f}")
    print(f"Activation Up-crossings:      {n_up}")
    print(f"Repolarization Down-crossings: {n_down}")
    print("-" * 65)

    # Replicate original structure condition: if(n_up == 2 & n_down == 2)
    # Note: bitwise & with lengths evaluated as integers mimics the original statement logic
    mapping_will_pass = (n_up == 2 and n_down == 2)

    if not mapping_will_pass:
        print("⚠️  VALIDATION FAILURE STATUS: WILL TURN WHITE ON 3D MAP")
        print("   Reason: Your 3D map pipeline explicitly requires EXACTLY 2 upstrokes")
        print(f"   and 2 downstrokes. This trace has Up: {n_up}, Down: {n_down}.")
        print("   The mapping code will reject this and assign an APD of 0.0 ms.")
    else:
        print("✅ VALIDATION SUCCESS STATUS: WILL COLOR CODES ON 3D MAP")
        print("   Reason: Exactly 2 upstrokes and 2 downstrokes detected.")

    # ─── 5. Calculate and Evaluate Timing & Drift ───────────────────────────
    print("-" * 65)
    print(f"Assumed Frame Step (Original Code): 4.000 ms/frame")
    print(f"Actual Linspace Frame Step (Plotting): {calculated_dt:.3f} ms/frame")
    
    max_loops = min(n_up, n_down)
    original_map_apds = []
    
    if max_loops == 0:
        print("\n❌ Cannot calculate pairings: Missing upstroke or downstroke crossing entirely.")
    else:
        for i in range(max_loops):
            up_idx = up_crossings[i]
            down_idx = down_crossings[i]
            
            # Match original 3D map hardcoded multiplier math
            orig_apd = (down_idx - up_idx) * 4.0
            original_map_apds.append(orig_apd)
            
            # Match visualization linspace math
            val_apd = time_ms[down_idx] - time_ms[up_idx]
            drift = abs(orig_apd - val_apd)
            
            print(f"\n[Beat {i+1}] (Frame Alignment: Up {up_idx} -> Down {down_idx})")
            print(f"  • 3D Map Logic APD (Hardcoded * 4.0):  {orig_apd:.1f} ms")
            print(f"  • Visual Linspace APD (Plot Scale):      {val_apd:.1f} ms")
            if drift > 0.01:
                print(f"  • ⚠️  Timing Discrepancy (Drift):        {drift:.4f} ms")

        if mapping_will_pass:
            final_map_value = (original_map_apds[0] + original_map_apds[1]) / 2.0
            print("-" * 65)
            print(f"➡️ Final Mapped APD Color Value on 3D Mesh: {final_map_value:.1f} ms")

    # ─── 6. Clean, Context-Aware Validation Plots ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel 1: Repolarization Focus (SavGol filter + Down-crossings)
    axes[0].plot(time_ms, trace_norm, color="#b0b0b0", linewidth=1.0, alpha=0.7, label="Normalized Trace")
    axes[0].plot(time_ms, trace_SG_norm, color="green", linewidth=1.5, label="SavGol Filtered (Used for Downstroke)")
    axes[0].axhline(APD_threshold, color="orange", linewidth=1.2, linestyle="--", label="10% APD Threshold")
    
    # Only mark down-crossings visually on the downstroke tracking plot
    for idx, down in enumerate(down_crossings):
        axes[0].axvline(time_ms[down], color="blue", linewidth=1.5, linestyle="-.", label="Down-crossing" if idx == 0 else "")
        axes[0].plot(time_ms[down], APD_threshold, 'bo', markersize=6)

    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Triangle {filenum:04d} ({recnum.upper()}) | Pipeline: {version.upper()}")
    axes[0].grid(True, alpha=0.2)

    # Panel 2: Activation Focus (Moving Average + Up-crossings)
    axes[1].plot(time_ms, trace_norm, color="#b0b0b0", linewidth=1.0, alpha=0.7)
    axes[1].plot(time_ms, trace_mn_norm, color="black", linewidth=1.5, label="Moving Average (Used for Upstroke)")
    axes[1].axhline(APD_threshold, color="orange", linewidth=1.2, linestyle="--")
    
    # Only mark up-crossings visually on the upstroke tracking plot
    for idx, up in enumerate(up_crossings):
        axes[1].axvline(time_ms[up], color="red", linewidth=1.5, label="Up-crossing" if idx == 0 else "")
        axes[1].plot(time_ms[up], APD_threshold, 'ro', markersize=6)

    # Annotate calculated map logic values over spans if validation passed
    if mapping_will_pass:
        for i in range(2):
            mid_time = (time_ms[up_crossings[i]] + time_ms[down_crossings[i]]) / 2.0
            axes[1].annotate(
                f"Map APD: {original_map_apds[i]:.1f} ms",
                xy=(mid_time, APD_threshold + 0.08),
                color="purple", weight="bold", ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="purple", lw=0.5)
            )

    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    
    # Save image systematically
    output_filename = f"validation_report_{version}_{recnum}_tri{filenum:04d}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\n📊 Validation plot saved as: {output_filename}")
    plt.show()

    return original_map_apds

# Run validation on specific channel
validate_apd10(version="old", recnum="rec02", filenum=306)