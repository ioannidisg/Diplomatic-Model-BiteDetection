from keras.layers import Dense, MaxPooling1D, LSTM, Dropout, Conv1D, TimeDistributed, BatchNormalization
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
# from dm import WindowsDataModule #1
from data_module import WindowsDataModule  #2
from utils.evaluation import extract_detected_times, calculate_f1_custom ,extract_detected_times_clustermax
from utils.plotting import plot_bite_intervals
from scipy.signal import find_peaks


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
    # build model
    model = baseline_train()

    # data
    dm = WindowsDataModule()
    dm.setup(split="global_by_session", test_ratio=0.2, seed=42)
    print("\n=== TEST COMPOSITION CHECK (FIC vs FreeFIC) ===")

    gt_n = 0 if dm.ground_truth_test is None else dm.ground_truth_test.shape[0]
    print("ground_truth_test intervals:", gt_n)
    print("X_test windows:", dm.X_test.shape[0])
    print("y_test positives:", int(dm.y_test.sum()))

    # train on TRAIN only (balanced)
    dataloader = dm.get_balanced_dataloader()

    # ------------------------------------------------------------
    # (NEW) Ensure checkpoints dir exists + print exact save location
    # ------------------------------------------------------------
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_full.weights.h5")

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss',
        mode='min',
        verbose=1
    )

    # ------------------------------------------------------------
    # (NEW) Class weights (first controlled change)
    # Note: because you already use a balanced dataloader, this is optional,
    # but we add it as the single next lever we agreed on.
    # If you want to disable it, set class_weight=None and remove from fit().
    # ------------------------------------------------------------
    pos = int(np.sum(dm.y_train == 1))
    neg = int(np.sum(dm.y_train == 0))
    w1 = float(min(200.0, neg / max(pos, 1)))
    class_weight = {0: 1.0, 1: w1}
    print(f"=== CLASS WEIGHTS === w0=1.0  w1={w1:.2f}  (neg={neg}, pos={pos})")

    history = model.fit(
        dataloader,
        epochs=5,
        steps_per_epoch=1000,
        callbacks=[ckpt],
        #class_weight=class_weight  # <- αν δεν το θες, σβήστο
    )

    # ------------------------------------------------------------
    # (NEW) Verify checkpoint file exists, then load
    # ------------------------------------------------------------
    if not os.path.exists(ckpt_path):
        print("[WARN] Checkpoint file not found after training:", os.path.abspath(ckpt_path))
        print("[WARN] That means it was not saved (or CWD is not what you expect).")
    else:
        print("✅ Checkpoint saved:", os.path.abspath(ckpt_path))
        model.load_weights(ckpt_path)

    # evaluate on explicit TEST set
    if dm.X_test.shape[0] == 0:
        print("[WARN] Empty X_test — adjust split/test_ratio")
        return

    print("=== DATA SPLIT CHECK ===")
    print("X_train shape:", dm.X_train.shape)
    print("X_test shape :", dm.X_test.shape)

    print("=== OFFSET SANITY CHECK 1 (index bounds) ===")
    mx = max(dm.window_starts_test) if len(dm.window_starts_test) else -1
    print("max window start:", mx)
    print("timestamps_test length:", len(dm.timestamps_test))
    print("max window end:", mx + dm.window_size - 1)

    assert mx >= 0, "No test windows."
    assert mx + dm.window_size - 1 < len(dm.timestamps_test), "Window end index out of bounds!"

    print("=== OFFSET SANITY CHECK 2 (window-end time monotonic) ===")
    ends = np.array([dm.timestamps_test[s + dm.window_size - 1] for s in dm.window_starts_test], dtype=np.float64)
    drops = np.where(np.diff(ends) < 0)[0] # ελέγχει ότι συνέχεια ο χρονος αυξάνεται
    print("window_end time drops:", len(drops))
    assert len(drops) == 0, "Non-monotonic window end times -> offset/time stitching issue!"

    y_pred = model.predict(dm.X_test)

    probs = y_pred.reshape(-1)
    ytrue = dm.y_test.reshape(-1).astype(int)

    ws = np.asarray(dm.window_starts_test, dtype=int)
    t_end = dm.timestamps_test[ws + dm.window_size - 1].astype(np.float64)
    mask_hard = (ytrue == 0) & (probs >= 0.99)
    hard_times = t_end[mask_hard]

    gt = dm.ground_truth_test.astype(np.float64)  # shape (N,2)
    gt_start = gt[:, 0]
    gt_end = gt[:, 1]

    def min_dist_to_intervals(t):
        inside = (t[:, None] >= gt_start[None, :]) & (t[:, None] <= gt_end[None, :])
        dist_start = np.abs(t[:, None] - gt_start[None, :])
        dist_end = np.abs(t[:, None] - gt_end[None, :])
        dist = np.minimum(dist_start, dist_end)
        dist[inside] = 0.0
        return dist.min(axis=1)

    print("\n=== HARD NEG DIAGNOSTIC (NEG prob >= 0.99) ===")

    if hard_times.size == 0:
        print("No NEG windows with prob >= 0.99")
    else:
        d = min_dist_to_intervals(hard_times)
        print("count:", hard_times.size)
        for thr in [0.0, 1.0, 2.0, 5.0, 10.0]:
            print(f"within ±{thr:>4.1f}s of GT:", f"{100.0 * np.mean(d <= thr):.2f}%")
        print(
            "min / median / mean / max dist (s):",
            f"{d.min():.3f}",
            f"{np.median(d):.3f}",
            f"{d.mean():.3f}",
            f"{d.max():.3f}",
        )

    pos_arr = probs[ytrue == 1]
    neg_arr = probs[ytrue == 0]

    def stats(name, arr):
        if arr.size == 0:
            print(f"{name}: EMPTY")
            return
        p = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        print(f"{name}: n={arr.size} mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  pct[1,5,10,25,50,75,90,95,99] = {np.round(p, 4)}")
        #pct[99] = 0.95 --> Το 99% prob < 0.95

    print("\n=== PROBABILITY DISTRIBUTION (TEST) ===")
    stats("POS (y=1)", pos_arr)
    stats("NEG (y=0)", neg_arr)

    bins = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89, 0.95, 0.99, 1.0])
    pos_hist, _ = np.histogram(pos_arr, bins=bins)
    neg_hist, _ = np.histogram(neg_arr, bins=bins)

    print("\nBins:", bins)
    print("POS hist:", pos_hist)
    print("NEG hist:", neg_hist)

    thresholds = [0.95, 0.89, 0.80, 0.73, 0.50, 0.30, 0.10]
    print("\n=== THRESHOLD CHECK (window-level, not peaks) ===")
    for th in thresholds:
        pos_keep = (pos_arr >= th).mean() if pos_arr.size else 0.0
        neg_keep = (neg_arr >= th).mean() if neg_arr.size else 0.0
        print(f"th={th:>4}: POS>=th {pos_keep * 100:6.2f}% | NEG>=th {neg_keep * 100:6.2f}%")

    print("=== PROBABILITY HISTOGRAM (TEST) ===")
    bins_lin = np.linspace(0, 1, 11)
    hist, edges = np.histogram(y_pred, bins=bins_lin)
    for i in range(len(hist)):
        print(f"{edges[i]:.1f}–{edges[i + 1]:.1f}: {hist[i]}")

    print("=== POSITIVE WINDOW PROBS ===")
    pos_probs = y_pred[dm.y_test == 1]
    hist_p, edges_p = np.histogram(pos_probs, bins=bins_lin)
    for i in range(len(hist_p)):
        print(f"{edges_p[i]:.1f}–{edges_p[i + 1]:.1f}: {hist_p[i]}")

    probs_tr = model.predict(dm.X_train, verbose=0).reshape(-1)
    ytr = dm.y_train.reshape(-1).astype(int)

    pos_tr = probs_tr[ytr == 1]
    neg_tr = probs_tr[ytr == 0]

    print("\n=== TRAIN PROBABILITY DISTRIBUTION ===")
    stats("TRAIN POS (y=1)", pos_tr)
    stats("TRAIN NEG (y=0)", neg_tr)

    # y_bin = postprocess_predictions(
    #     y_pred,
    #     threshold=0.89,
    #     min_distance_sec=2.0,
    #     stride=dm.stride,
    #     fs=100
    # )

    print("Predicting on:", dm.X_test.shape)
    print("=== WINDOW ALIGNMENT CHECK ===")
    print("X_test windows:", dm.X_test.shape[0])
    print("window_starts_test:", len(dm.window_starts_test))

    detected_intervals = extract_detected_times_clustermax(
        probs=y_pred,
        window_starts=dm.window_starts_test,
        timestamps=dm.timestamps_test,
        window_size=dm.window_size,
        threshold=0.9,
        K=4,
        min_distance_sec=5.0,
        stride=dm.stride,
        fs=100
    )


    print("=== TIME AXIS CHECK ===")
    drops = np.where(np.diff(dm.timestamps_test) < 0)[0]
    print("Time drops in timestamps_test:", len(drops))

    print("=== EVENT COUNT CHECK ===")
    print("Detected events:", len(detected_intervals))
    print("GT events:", len(dm.ground_truth_test))

    # stride-based epsilons (ONLY evaluation)
    eps_step = dm.stride / 100.0  # seconds per window step (fs=100)
    eps_half = eps_step / 2.0  # half-step tolerance

    f1_half = calculate_f1_custom(detected_intervals, dm.ground_truth_test, epsilon=eps_half)
    f1_step = calculate_f1_custom(detected_intervals, dm.ground_truth_test, epsilon=eps_step)
    f1_25 = calculate_f1_custom(detected_intervals, dm.ground_truth_test, epsilon=2.5)

    print(f"[FULL] F1 eps=half  : {f1_half:.4f}  (half={eps_half:.3f}s)")
    print(f"[FULL] F1 eps=step  : {f1_step:.4f}  (step={eps_step:.3f}s)")
    print(f"[FULL] F1 eps=2.5   : {f1_25:.4f} (step={2.5:.3f}s)")


    os.makedirs("plots", exist_ok=True)
    plot_bite_intervals(detected_intervals, dm.ground_truth_test, save_path="plots/bite_vs_gt_full_test.png")

def train_LOSO():

    os.makedirs("plots/loss_10epochs", exist_ok=True)
    os.makedirs("plots/accuracy_10epochs", exist_ok=True)

    num_epochs = 4

    subject_ids = list(range(1, 13))  # 12 subjects
    f1_scores = []
    final_train_losses = []
    final_train_accuracies = []

    for subject_id in subject_ids:
        print(f"\n Fold for subject {subject_id}")

        # Setup data
        data_module = WindowsDataModule()  # 2
        data_module.setup_LOSO(test_subject_id=subject_id)
       # dataloader = data_module.get_dataloader_LOSO()
        dataloader = data_module.get_balanced_dataloader_LOSO();

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./checkpoints/subject_{subject_id}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor='loss',
            mode='min',
            verbose=1
        )


        # Build new model
        model = baseline_train()

        # Train model
        #history = model.fit(dataloader, epochs=num_epochs)
        history = model.fit(dataloader, epochs=num_epochs, steps_per_epoch=2000, callbacks=[checkpoint_callback])
        model.load_weights(f"./checkpoints/subject_{subject_id}.weights.h5")

        train_loss_last = history.history['loss'][-1]
        train_acc_last = history.history['accuracy'][-1]
        final_train_losses.append(train_loss_last)
        final_train_accuracies.append(train_acc_last)

        # Predict on test set
        y_pred = model.predict(data_module.X_test)
        # y_pred_bin = (y_pred > 0.89).astype(int)
        y_pred_bin = postprocess_predictions(
            y_pred,
            threshold=0.80,
            min_distance_sec=2.0,
            stride=data_module.stride,  # π.χ. 250
            fs=100  # sampling rate σου
        )
        print("y_pred_bin.sum():", np.sum(y_pred_bin))  # Πόσα '1' προβλέφθηκαν
        print("y_pred range:", y_pred.min(), "-", y_pred.max())  # Πόσο χαμηλές είναι οι πιθανότητες
        print("y_pred mean:", y_pred.mean())

        # debugging
        # Compute F1 score
        # f1 = f1_score(data_module.y_test, y_pred_bin)
        # f1_scores.append(f1)

        timestamps_test = data_module.timestamps_test
        window_starts_test = data_module.window_starts_test
        ground_truth_intervals = data_module.ground_truth_test

        detected_intervals = extract_detected_times(
            y_pred_bin,
            window_starts_test,
            timestamps_test,
            window_size=500
        )


        f1 = calculate_f1_custom(detected_intervals, ground_truth_intervals, epsilon=0.2)
        f1_scores.append(f1)

        print(f"F1 Score for subject {subject_id}: {f1:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Loss')
        plt.title(f"Loss per Epoch, Subject: {subject_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/loss_10epochs/loss_subject_{subject_id}.png")
        plt.close()

       # plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.title(f"Accuracy per Epoch, Subject: {subject_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/accuracy_10epochs/acc_subject_{subject_id}.png")
        plt.close()
        #plt.show()

        # Print final training loss and accuracy
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        print(f"Final Training Loss: {train_loss:.4f}")
        print(f"Final Training Accuracy: {train_accuracy:.4f}")
        plot_path = f"plots/bite_vs_gt_subject_{subject_id}.png"
        plot_bite_intervals(detected_intervals, ground_truth_intervals, save_path=plot_path)

    # Overall results
    print("\LOSO Complete")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    print(f"F1 Scores per subject: {[f'{s:.4f}' for s in f1_scores]}")

    avg_train_loss = np.mean(final_train_losses)
    std_train_loss = np.std(final_train_losses, ddof=1)  # sample std
    print(f"Average FINAL Training Loss across folds: {avg_train_loss:.4f} (std: {std_train_loss:.4f})")

import numpy as np

def postprocess_predictions(y_pred, threshold=0.5, min_distance_sec=2.0, stride=250, fs=100):
  
    p = y_pred.reshape(-1).astype(float)

    # 1) thresholding
    p[p < threshold] = 0.0

    # windows per second
    frame_rate = fs / stride
    distance = max(1, int(np.ceil(min_distance_sec * frame_rate)))

    peaks = []
    i = 1
    n = len(p)

    # 2) plateau-safe local maxima detection
    while i < n - 1:
        if p[i] <= 0:
            i += 1
            continue

        l = i
        while l > 0 and p[l-1] == p[i]:
            l -= 1
        r = i
        while r < n - 1 and p[r+1] == p[i]:
            r += 1

        left = p[l-1] if l > 0 else -np.inf
        right = p[r+1] if r < n-1 else -np.inf

        if p[i] > left and p[i] > right:
            peaks.append((l + r) // 2)

        i = r + 1

    if not peaks:
        return np.zeros_like(p, dtype=int)

    peaks = np.array(peaks, dtype=int)

    # 3) refractory: keep strongest within +/- distance
    order = np.argsort(p[peaks])[::-1]
    peaks_sorted = peaks[order]

    selected = []
    suppressed = np.zeros(n, dtype=bool)

    for idx in peaks_sorted:
        if suppressed[idx]:
            continue
        selected.append(idx)
        lo = max(0, idx - distance)
        hi = min(n, idx + distance + 1)
        suppressed[lo:hi] = True

    y_bin = np.zeros_like(p, dtype=int)
    y_bin[selected] = 1
    return y_bin




if __name__ == "__main__":
   train_full()
 #  train_LOSO()
