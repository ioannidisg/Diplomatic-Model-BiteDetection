import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_module import WindowsDataModule
from model_genFixed import baseline_train
from utils.evaluation import extract_detected_times_paper_localmax_2  # ή _paper_localmax

# =========================
# F1 helper (όπως το δικό σου)
# =========================
def f1_custom_stats(detected_intervals, ground_truth_intervals, epsilon):
    TP = 0
    FP = 0
    matched_gt = set()

    for det_start, det_end in detected_intervals:
        matched = False
        for j, (gt_start, gt_end) in enumerate(ground_truth_intervals):
            if j in matched_gt:
                continue
            if det_start <= gt_end + epsilon and det_end >= gt_start - epsilon:
                TP += 1
                matched_gt.add(j)
                matched = True
                break
        if not matched:
            FP += 1

    FN = len(ground_truth_intervals) - len(matched_gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, TP, FP, FN, precision, recall


def predict_probs_for_session(model, dm, s, CHUNK=1024):
    """Return (starts, probs, ts_plot) for a FIC session."""
    low = (s["signal_low"] - dm.scaler_params["mean_low"]) / (dm.scaler_params["scale_low"] + 1e-8)
    high = (s["signal_high"] - dm.scaler_params["mean_high"]) / (dm.scaler_params["scale_high"] + 1e-8)
    sensors = np.hstack([low, high]).astype(np.float32)

    starts = np.arange(0, len(sensors) - dm.window_size + 1, dm.stride_fic, dtype=np.int64)
    if len(starts) == 0:
        return starts, np.array([], dtype=np.float32), (s["ts_eff"] + s.get("time_offset", 0.0))

    X = np.stack([sensors[i:i+dm.window_size] for i in starts], axis=0)
    probs = []
    for j in range(0, len(X), CHUNK):
        probs.append(model.predict(X[j:j+CHUNK], verbose=0).reshape(-1))
    probs = np.concatenate(probs).astype(np.float32)

    ts_plot = s["ts_eff"] + s.get("time_offset", 0.0)
    return starts, probs, ts_plot


def get_fic_ground_truth_intervals(s):
    """
    Βγάζει ground truth σε μορφή intervals (N,2).
    Χρησιμοποιεί ΜΟΝΟ events_eff (+offset αν υπάρχει).
    """
    events = s["events_eff"] + s.get("time_offset", 0.0)

    if events.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    events = np.asarray(events, dtype=np.float64)

    if events.ndim == 1:
        # timestamps
        gt = np.stack([events, events], axis=1)
        return gt

    if events.ndim == 2 and events.shape[1] == 2:
        return events

    # fallback
    raise ValueError(f"Unexpected events_eff shape: {events.shape}")


def make_fic_plot(s, ts_plot, ax_x, gt_intervals, det_times, out_path, title=""):
    plt.figure(figsize=(18, 4))

    # GT as shaded spans
    for (a, b) in gt_intervals:
        # αν είναι (t,t) θα είναι πολύ λεπτό -> βάλε μικρό πάχος να φαίνεται
        if abs(b - a) < 1e-6:
            plt.axvspan(a - 0.1, a + 0.1, color="grey", alpha=0.2)
        else:
            plt.axvspan(a, b, color="grey", alpha=0.2)

    # signal
    plt.plot(ts_plot, ax_x, linewidth=1.0, label="a_x")

    # detections
    if len(det_times) > 0:
        y_det = np.interp(det_times, ts_plot, ax_x)
        plt.scatter(det_times, y_det, marker="x", s=60, color="red", label="b_i")

    plt.title(title)
    plt.xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    CKPT_PATH = os.path.join("checkpoints", "new.weights.h5")
    OUT_DIR = "new_debug_plots"
    os.makedirs(OUT_DIR, exist_ok=True)

    #THRESHOLDS = [ 0.35, 0.40, 0.50, 0.70, 0.89]
    THRESHOLDS = [0.76]
    MIN_DIST_SEC = 2.0
    K = 2                    # consecutive gate (1/2/3 δοκίμασε)
    EVAL_EPS = [1.0, 1.5, 2.0, 2.5]
    CHUNK = 1024

    dm = WindowsDataModule()
    dm.setup(split="global_by_session", test_ratio=0.2, seed=42)

    fic_test = [s for s in dm.test_sessions if s["type"] == "fic"]
    print(f"FIC test sessions: {len(fic_test)}")
    
    model = baseline_train()
    model.load_weights(CKPT_PATH)
    print("Loaded weights:", CKPT_PATH)

    cache = []
    for s in fic_test:
        starts, probs, ts_plot = predict_probs_for_session(model, dm, s, CHUNK=CHUNK)
        gt = get_fic_ground_truth_intervals(s)
        ax_x = s["signal_raw"][:, 0].astype(np.float64)  # raw x axis
        cache.append((s, starts, probs, ts_plot, gt, ax_x))

    total_gt = int(sum(len(item[4]) for item in cache))
    print(f"[GT TOTAL] TOTAL GT BITES (FIC test): {total_gt}")

    for thr in THRESHOLDS:
        detected_intervals_global = []
        gt_global = []

        det_count = 0

        for (s, starts, probs, ts_plot, gt, ax_x) in cache:
            gt_global.append(gt)

            if len(probs) == 0:
                continue

            det_times = extract_detected_times_paper_localmax_2(
                probs=probs,
                window_starts=starts,
                timestamps=ts_plot,
                window_size=dm.window_size,
                threshold=thr,
                min_distance_sec=MIN_DIST_SEC,
                stride=dm.stride_fic,
                fs=dm.fs,
                K=K
            )
            det_count += len(det_times)

            # global detections as point-intervals
            for t in det_times:
                detected_intervals_global.append((float(t), float(t)))

            # save plot for this session+threshold
            out_path = os.path.join(OUT_DIR, f"fic_idx{s['idx']}_thr{thr:.2f}.png")
            title = f"FIC idx={s['idx']} thr={thr:.2f} K={K}"
            make_fic_plot(s, ts_plot, ax_x, gt, det_times, out_path, title=title)

        gt_global = np.vstack(gt_global) if len(gt_global) else np.empty((0, 2), dtype=np.float64)

        print(f"\n--- threshold={thr:.2f} | detections={det_count} | plots -> {OUT_DIR}/ ---")
        for eps in EVAL_EPS:
            f1, TP, FP, FN, P, R = f1_custom_stats(detected_intervals_global, gt_global, epsilon=eps)
            print(f"eps={eps:.1f}s | TP:{TP} FP:{FP} FN:{FN} | P:{P:.2f} R:{R:.2f} | F1:{f1:.4f}")


if __name__ == "__main__":
    main()
