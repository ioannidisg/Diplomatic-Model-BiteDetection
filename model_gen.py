from keras.layers import Dense, MaxPooling1D, LSTM, Dropout, Conv1D, TimeDistributed, BatchNormalization
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import f1_score ,classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
# from dm import WindowsDataModule #1
from data_moduleFixed import WindowsDataModule  #2
from data_generator2 import BalancedWindowSequence
from utils.evaluation import extract_detected_times, calculate_f1_custom ,extract_detected_times_clustermax,extract_detected_times_paper_localmax
from scipy.signal import find_peaks
from plot import plot_paper_style_fic_session

def baseline_train():
    this_optimizer = RMSprop()
    model = Sequential()
    model.add(
        Conv1D(
            filters=32, kernel_size=5, padding='same',
            activation='relu', input_shape=(500, 6)
            )
        )
    model.add(BatchNormalization())

    model.add(MaxPooling1D())

    model.add(
        Conv1D(
            filters=64, kernel_size=3, # filters είναι η νευρώνες(units)
            padding='same', activation='relu'
            )
        )
    model.add(BatchNormalization())

    model.add(MaxPooling1D())

    model.add(
        Conv1D(
            filters=128, kernel_size=3,
            padding='same', activation='relu'
            )
        )
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(0.5)) # σβήνει το 50% των νευρώνων (τυχαία) για να μην έχω overfitting
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=this_optimizer, metrics=['accuracy']
        )
   # model.summary()
    return model


def baseline_predict():
    this_optimizer = RMSprop()
    model = Sequential()
    model.add(
        Conv1D(
            filters=32, kernel_size=5, padding='same',
            activation='relu', input_shape=(None, 6)
            )
        )
    model.add(BatchNormalization())

    model.add(MaxPooling1D())

    model.add(
        Conv1D(
            filters=64, kernel_size=3,
            padding='same', activation='relu'
            )
        )
    model.add(BatchNormalization())

    model.add(MaxPooling1D())

    model.add(
        Conv1D(
            filters=128, kernel_size=3,
            padding='same', activation='relu'
            )
        )
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed((Dropout(0.5))))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(
        loss='binary_crossentropy',
        optimizer=this_optimizer, metrics=['accuracy']
        )
    return model
def train_full():
    import os, gc
    import numpy as np
    import tensorflow as tf

    # ============================
    # CONFIG
    # ============================
    EPOCHS = 50
    STEPS_PER_EPOCH = 1500
    THRESHOLD = 0.89
    POST_MIN_DISTANCE_SEC = 2.0
    LABEL_EPS_FIC = 0.1
    EVAL_EPSILONS = [1.0,1.5,2.0 ,2.5]
    CHUNK = 1024

    # ============================
    # F1 + stats helper
    # ============================
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

    # ============================
    # 1) Build model
    # ============================
    model = baseline_train()

    # ============================
    # 2) Data setup
    # ============================
    dm = WindowsDataModule()
    dm.setup(split="global_by_session", test_ratio=0.2, seed=42)

    # ============================
    # 3) Generator
    # ============================
    train_gen = BalancedWindowSequence(
        sessions=dm.train_sessions,
        batch_size=128,
        window_size=dm.window_size,
        stride=dm.stride_fic,             # 5
        stride_freefic=dm.stride_freefic, # 100
        scaler_params=dm.scaler_params,
        epsilon=LABEL_EPS_FIC,
        ratio=1
    )

    # ============================
    # 4) Training
    # ============================
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_fixed.weights.h5")

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        mode="min",
        verbose=1
    )

    model.fit(
        train_gen,
        epochs=EPOCHS,
        #steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[ckpt],
        verbose=1
    )

    model.load_weights(ckpt_path)

    # ============================
    # 5) STREAMING EVALUATION
    # ============================
    print("\n=== STARTING GLOBAL EVALUATION ===")

    detected_intervals = []
    ground_truth = []

    probs_all = []
    ytrue_all = []

    for s in dm.test_sessions:
        # normalize
        low = (s["signal_low"] - dm.scaler_params["mean_low"]) / dm.scaler_params["scale_low"]
        high = (s["signal_high"] - dm.scaler_params["mean_high"]) / dm.scaler_params["scale_high"]
        sensors = np.hstack([low, high]).astype(np.float32)

        is_fic = (s["type"] == "fic")
        stride = dm.stride_fic if is_fic else dm.stride_freefic
        # --- DEBUG: ground-truth shapes/counts per session ---
        if s["type"] == "fic":
            ev = s.get("events_shifted", None)
            if ev is None:
                print(f"[GT DEBUG] FIC idx={s.get('idx')} events_shifted = None")
            else:
                ev_np = np.asarray(ev)
                print(f"[GT DEBUG] FIC idx={s.get('idx')} events_shifted shape={ev_np.shape} dtype={ev_np.dtype}")
        else:
            ev = s.get("events", None)
            if ev is None:
                print(f"[GT DEBUG] FreeFIC idx={s.get('idx')} events(meals)=None")
            else:
                ev_np = np.asarray(ev)
                print(f"[GT DEBUG] FreeFIC idx={s.get('idx')} meals shape={ev_np.shape} dtype={ev_np.dtype}")
# -----------------------------------------------------

        starts = np.arange(0, len(sensors) - dm.window_size + 1, stride)
        if len(starts) == 0:
            continue

        X = []
        y_batch = []
        events = s["events"]

        for start in starts:
            end = start + dm.window_size
            delay = s["ts"][0] - s["ts_eff"][0]
            t_end = s["ts_eff"][end - 1] + delay  # shift t_end to match original event timebase

            label = 0
            if is_fic:
                if len(events) > 0:
                    label = 1 if np.any(np.abs(t_end - events[:, 1]) <= LABEL_EPS_FIC) else 0
            else:
                if len(events) > 0:
                    inside = np.any((t_end >= events[:, 0]) & (t_end <= events[:, 1]))
                    label = -1 if inside else 0

            X.append(sensors[start:end])
            y_batch.append(label)

        X = np.asarray(X, dtype=np.float32)

        probs = []
        for j in range(0, len(X), CHUNK):
            probs.append(model.predict(X[j:j + CHUNK], verbose=0).reshape(-1))
        probs = np.concatenate(probs)

        probs_all.extend(probs.tolist())
        ytrue_all.extend(y_batch)

        ts = s["ts_eff"] + s["time_offset"]

        # ---------- FIC ----------
        if is_fic:
            bite_times = extract_detected_times_paper_localmax(
                probs=probs,
                window_starts=starts,
                timestamps=ts,
                window_size=dm.window_size,
                threshold=THRESHOLD,
                min_distance_sec=POST_MIN_DISTANCE_SEC,
                stride=dm.stride_fic,
                fs=dm.fs
            )
            
            detected_intervals.extend([(float(t), float(t)) for t in bite_times])
         
            if "events_shifted" in s and s["events_shifted"] is not None and len(s["events_shifted"]) > 0:
                ground_truth.append(s["events_shifted"])
           
            plot_paper_style_fic_session(s=s,probs=probs,starts=starts,dm=dm,detected_times=bite_times,LABEL_EPS_FIC=LABEL_EPS_FIC,title=f"FIC debug idx={s['idx']}" )

        # ---------- FreeFIC ----------
        else:
            bite_times = extract_detected_times_paper_localmax(
                probs=probs,
                window_starts=starts,
                timestamps=ts,
                window_size=dm.window_size,
                threshold=THRESHOLD,
                min_distance_sec=POST_MIN_DISTANCE_SEC,
                stride=dm.stride_freefic,
                fs=dm.fs
            )

            delay = s["ts"][0] - s["ts_eff"][0]
            meals_eff_global = (s["events"] - delay) + s["time_offset"]

            for t in bite_times:
                if not np.any((t >= meals_eff_global[:, 0]) & (t <= meals_eff_global[:, 1])):
                    detected_intervals.append((float(t), float(t)))

    if ground_truth:
        ground_truth = np.vstack(ground_truth)
    else:
        ground_truth = np.empty((0, 2))

    # ============================
    # 6) PROBABILITY DISTRIBUTIONS (exclude -1)
    # ============================
    print("\n=== PROBABILITY DISTRIBUTION (VALID WINDOWS) ===")

    probs_all = np.asarray(probs_all, dtype=np.float64)
    ytrue_all = np.asarray(ytrue_all, dtype=np.int32)

    valid = (ytrue_all != -1)
    probs_v = probs_all[valid]
    ytrue_v = ytrue_all[valid]

    pos_arr = probs_v[ytrue_v == 1]
    neg_arr = probs_v[ytrue_v == 0]

    def stats(name, arr):
        if arr.size == 0:
            print(f"{name}: EMPTY")
            return
        p = np.percentile(arr, [1,5,10,25,50,75,90,95,99])
        print(f"{name}: n={arr.size} mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  pct[1,5,10,25,50,75,90,95,99] = {np.round(p,4)}")

    stats("POS (y=1)", pos_arr)
    stats("NEG (y=0)", neg_arr)

    bins = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89, 0.95, 0.99, 1.0])
    pos_hist, _ = np.histogram(pos_arr, bins=bins)
    neg_hist, _ = np.histogram(neg_arr, bins=bins)
    print("\nBins:", bins)
    print("POS hist:", pos_hist)
    print("NEG hist:", neg_hist)

    # ============================
    # 7) FINAL GLOBAL EVENT METRICS
    # ============================
    print("\n=== GLOBAL EVENT METRICS ===")
    for eps in EVAL_EPSILONS:
        f1, TP, FP, FN, P, R = f1_custom_stats(detected_intervals, ground_truth, epsilon=eps)
        print(f"eps={eps:.1f}s | TP:{TP} FP:{FP} FN:{FN} | P:{P:.2f} R:{R:.2f} | F1:{f1:.4f}")

    tf.keras.backend.clear_session()
    gc.collect()

    return detected_intervals, ground_truth


if __name__ == "__main__":
   train_full()
 #  train_LOSO()






