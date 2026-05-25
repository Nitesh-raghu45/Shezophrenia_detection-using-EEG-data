import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    print("=" * 60)
    print("  EEG Schizophrenia - Synthetic Dataset Generator")
    print("=" * 60)

    dataset_dir = Path("./dummy_dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create demographic.csv
    # Column names must be 'subject' and 'group' (or strip spacing if needed)
    demo_path = dataset_dir / "demographic.csv"
    subjects = [1, 2, 67, 68]
    groups = [0, 0, 1, 1]  # 0=HC, 1=SZ
    
    df_demo = pd.DataFrame({
        "subject": subjects,
        "group": groups
    })
    df_demo.to_csv(demo_path, index=False)
    print(f"Created: {demo_path}")

    # 2. Create mock CSVs for subjects (no header)
    # Col 0: trial, Col 1: sample, Col 2: id, Col 3: time, Cols 4-19: 16 channels
    n_channels = 16
    samples_per_trial = 1000
    trials = [1, 2]

    for subj, grp in zip(subjects, groups):
        file_path = dataset_dir / f"{subj}.csv"
        
        data_rows = []
        for trial_num in trials:
            # Generate a base sine wave for brain waves plus random noise
            t = np.linspace(0, 5, samples_per_trial)
            # Simulated alpha band (10Hz) and beta band (18Hz)
            signal_alpha = np.sin(2 * np.pi * 10 * t)
            signal_beta = 0.5 * np.sin(2 * np.pi * 18 * t)
            
            for s_idx in range(samples_per_trial):
                # Add some unique activity per channel
                ch_values = []
                for ch in range(n_channels):
                    # Introduce a slight phase shift per channel and some noise
                    phase = ch * 0.1
                    noise = np.random.normal(0, 0.1)
                    val = signal_alpha[s_idx] * np.cos(phase) + signal_beta[s_idx] * np.sin(phase) + noise
                    # SZ subjects get slightly higher amplitude noise to simulate abnormal EEG spikes
                    if grp == 1:
                        val += np.random.normal(0, 0.15)
                    ch_values.append(f"{val:.6f}")
                
                # Format row: trial, sample, id, time, [16 electrode values]
                row = [
                    str(trial_num),
                    str(s_idx),
                    str(subj),
                    f"{t[s_idx]:.4f}"
                ] + ch_values
                data_rows.append(row)
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data_rows)
        print(f"Created subject data: {file_path} ({len(data_rows)} rows)")

    print("\n[OK] Synthetic dataset created successfully in ./dummy_dataset")
    print("=" * 60)

if __name__ == "__main__":
    main()
