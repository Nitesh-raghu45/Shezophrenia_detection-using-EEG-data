import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_section("EEG Schizophrenia Detection - End-to-End Pipeline Orchestrator")
    
    # Paths
    project_root = Path(__file__).parent.resolve()
    dummy_dataset = project_root / "dummy_dataset"
    eeg_images = project_root / "eeg_images"
    features_cache = project_root / "features_cache"
    
    # 1. Clean previous runs
    print("[Clean] Cleaning previous generated files...")
    for folder in [dummy_dataset, eeg_images, features_cache, 
                   project_root / "model_output", 
                   project_root / "cnn_bilstm_output", 
                   project_root / "hybrid_output", 
                   project_root / "svm_output"]:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Removed: {folder}")
            
    # 2. Run generate_synthetic_data.py
    print_section("Step 1: Generating Synthetic EEG Dataset")
    subprocess.run([sys.executable, "generate_synthetic_data.py"], check=True)

    # 3. Run generate_spectrograms.py
    print_section("Step 2: Generating Mel-Spectrograms")
    subprocess.run([
        sys.executable, "generate_spectrograms.py",
        "--dataset", str(dummy_dataset),
        "--out", str(eeg_images),
        "--images_per_class", "20",  # Generate small number of images for testing
        "--workers", "2"
    ], check=True)

    # 4. Run train_efficientnet.py
    print_section("Step 3: Training EfficientNetB0 (Feature Extraction & Cached Training)")
    subprocess.run([
        sys.executable, "train_efficientnet.py",
        "--data", str(eeg_images),
        "--out", "model_output",
        "--epochs", "2",          # Keep epochs low for verification
        "--batch", "8",
        "--fine_tune", "0"        # Skip Phase 2 fine-tuning for quick check
    ], check=True)

    # 5. Run train_cnn_bilstm.py
    print_section("Step 4: Training CNN + BiLSTM (Sequence Feature Extraction & Cached Training)")
    subprocess.run([
        sys.executable, "train_cnn_bilstm.py",
        "--data", str(eeg_images),
        "--out", "cnn_bilstm_output",
        "--epochs", "2",
        "--batch", "8",
        "--fine_tune", "0"
    ], check=True)

    # 6. Run train_hybrid_model.py
    print_section("Step 5: Training Random Forest (Loading Shared Cache)")
    subprocess.run([
        sys.executable, "train_hybrid_model.py",
        "--data", str(eeg_images),
        "--out", "hybrid_output",
        "--clf", "rf",
        "--cv", "2"
    ], check=True)

    # 7. Run train_hybrid_svm.py
    print_section("Step 6: Training SVM (Loading Shared Cache)")
    subprocess.run([
        sys.executable, "train_hybrid_svm.py",
        "--data", str(eeg_images),
        "--out", "svm_output",
        "--clf_kernel", "rbf",
        "--cv", "2"
    ], check=True)

    # 8. Run make_ppt.py
    print_section("Step 7: Generating Project Presentation")
    subprocess.run([sys.executable, "make_ppt.py"], check=True)

    # 9. Run generate_report.py
    print_section("Step 8: Generating Project PDF Report")
    subprocess.run([sys.executable, "generate_report.py"], check=True)

    print_section("Pipeline Execution Complete!")
    print("All outputs generated successfully:")
    print(f" - Image outputs : {eeg_images}")
    print(f" - Feature cache : {features_cache}")
    print(f" - EfficientNet  : ./model_output")
    print(f" - CNN + BiLSTM  : ./cnn_bilstm_output")
    print(f" - Hybrid RF     : ./hybrid_output")
    print(f" - Hybrid SVM    : ./svm_output")
    print(" - Presentation  : ./EEG_Schizophrenia_Detection.pptx")
    print(" - PDF Report    : ./minor_project_report.pdf")
    print("=" * 80)

if __name__ == "__main__":
    main()
