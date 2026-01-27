import os
import numpy as np
import tensorflow as tf

# import το datamodule σου
from data_module import WindowsDataModule

# import τις συναρτήσεις που ήδη χρησιμοποιείς για postprocess + evaluation
# ΠΡΟΣΑΡΜΟΣΕ τα imports ώστε να δείχνουν στα δικά σου paths/αρχεία:
from model_gen import postprocess_predictions
from evaluation import extract_detected_times, calculate_f1_custom


def eval_only(
    model_path="saved_models/latest.keras",
    epsilon_match=0.2,
    fs=100,
    min_distance_sec=2.0,
    threshold_grid=(0.95, 0.90, 0.89, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50),
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Run training once and save the model there."
        )

    # 1) Load data
    dm = WindowsDataModule()
    dm.setup(split="global_by_session", test_ratio=0.2, seed=42)

    # 2) Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model_path}")

    # 3) Predict ONCE
    y_pred = model.predict(dm.X_test, verbose=0)
    print(f"✅ Predicted on X_test: {dm.X_test.shape}")

    # 4) Threshold grid (event-level)
    print("\n=== THRESHOLD GRID (event-level) ===")
    best_f1 = -1.0
    best_th = None

    for th in threshold_grid:
        y_bin = postprocess_predictions(
            y_pred,
            threshold=th,
            min_distance_sec=min_distance_sec,
            stride=dm.stride,
            fs=fs
        )

        detected_intervals = extract_detected_times(
            y_bin,
            dm.window_starts_test,
            dm.timestamps_test,
            window_size=dm.window_size
        )

        print(f"\n--- th={th} | detected={len(detected_intervals)} ---")
        f1 = calculate_f1_custom(
            detected_intervals,
            dm.ground_truth_test,
            epsilon=epsilon_match
        )

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\n✅ BEST threshold: {best_th} | BEST F1: {best_f1:.4f}")


if __name__ == "__main__":
    eval_only()
