import pickle as pkl
import numpy as np
import tensorflow as tf
from scipy.signal import firwin
from pathlib import Path
import json

from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import preprocess_acc_data, apply_filters


class WindowsDataModule:
    def __init__(self):
        self.pkl_path = "data/FIC.pkl"
        self.freefic_path = "data/FreeFIC_FreeFIC-heldout.pkl"

        self.batch_size = 128
        self.window_size = 500
        self.stride = 125
        self.sensor_type = "acc"  # key inside FIC sessions

        self.scaler_params = None

    # ============================================================
    # GLOBAL SPLIT SETUP (train/test by session OR per_subject)
    # ============================================================
    def setup(self, stage=None, split="global_by_session", test_ratio=0.2, seed=42):
        rng = np.random.RandomState(seed)

        # ---------- Load FIC ----------
        with open(self.pkl_path, "rb") as fh:
            dataset = pkl.load(fh)

        raw_data = dataset["signals_raw"]
        bite_gt = dataset["bite_gt"]
        subject_ids = dataset.get("subject_id", None)

        # ============================================================
        # 1) Collect eligible sessions (NO preprocessing yet)
        # ============================================================
        sessions = []  # each: {idx, subj, ts, sensors, bites, win}


        for i, session in enumerate(raw_data):
            acc_full = session[self.sensor_type]  # (N, 1+ch)
            timestamps = acc_full[:, 0]
            sensors = acc_full[:, 1:4]  # (N,3)
            bites = bite_gt[i]

            # eligibility: must produce >=1 window
            if len(sensors) < self.window_size:
                continue

            win = np.arange(0, len(sensors) - self.window_size + 1, self.stride, dtype=int)
            if win.size == 0:
                continue

            sessions.append({
                "idx": i,
                "subj": int(subject_ids[i]) if subject_ids is not None else None,
                "ts": timestamps,
                "sensors": sensors,
                "bites": bites,
                "win": win
            })

        if not sessions:
            raise ValueError("No eligible FIC sessions produced windows. Check window_size/stride.")

        # ============================================================
        # 2) Choose TEST sessions from eligible
        # ============================================================
        if split == "per_subject" and subject_ids is not None:
            subj2idx = defaultdict(list)
            for s in sessions:
                subj2idx[s["subj"]].append(s["idx"])

            test_idx = set()
            for sid, idxs in subj2idx.items():
                idxs = list(idxs)
                rng.shuffle(idxs)
                n_test = max(1, int(np.ceil(len(idxs) * test_ratio))) if len(idxs) > 1 else 1
                test_idx.update(idxs[:n_test])

        else:  # global_by_session
            idxs = [s["idx"] for s in sessions]
            rng.shuffle(idxs)
            n_test = max(1, int(np.ceil(len(idxs) * test_ratio)))
            test_idx = set(idxs[:n_test])

        # ============================================================
        # 3) PASS 1 — Global scaler stats (TRAIN ONLY)
        # ============================================================
        print("🔄 Calculating Global Scaler Stats on TRAIN only...")

        scaler_low = StandardScaler()
        scaler_high = StandardScaler()
        # FIC TRAIN
        for s in sessions:
            if s["idx"] in test_idx:
                continue

            low, high = apply_filters(s["sensors"])
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        # FreeFIC (optional but recommended): include in scaling stats
        with open(self.freefic_path, "rb") as fh:
            freefic = pkl.load(fh)
        free_raw = freefic["signals_raw"]

        for sess in free_raw:
            acc_full = sess["acc"]  # (N,4): [t, ax, ay, az]
            acc_free = acc_full[:, 1:4]
            low, high = apply_filters(acc_free)
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        self.scaler_params = {
            "mean_low": scaler_low.mean_.astype(np.float32),
            "scale_low": scaler_low.scale_.astype(np.float32),
            "mean_high": scaler_high.mean_.astype(np.float32),
            "scale_high": scaler_high.scale_.astype(np.float32),
        }

        print("✅ Global scaler stats ready.")
        print("   mean_low:", self.scaler_params["mean_low"])
        print("   scale_low:", self.scaler_params["scale_low"])
        l_ma = 25
        f_c = 1
        l_hp = 513
        f = 100

        ma = np.ones(l_ma, dtype=np.float32) / l_ma
        b_low = firwin(l_hp, f_c, pass_zero=True, fs=f).astype(np.float32)
        b_high = firwin(l_hp, f_c, pass_zero=False, fs=f).astype(np.float32)

        payload = {
            "ma": ma.tolist(),
            "b_low": b_low.tolist(),
            "b_high": b_high.tolist(),
            "mean_low": self.scaler_params["mean_low"].tolist(),
            "std_low": self.scaler_params["scale_low"].tolist(),
            "mean_high": self.scaler_params["mean_high"].tolist(),
            "std_high": self.scaler_params["scale_high"].tolist(),
        }

        json_file = Path("filters_acc_2.json")
        json_file.write_text(json.dumps(payload, indent=2))
        print(f"✅ Filters & scalers saved to: {json_file.resolve()}")
        # ============================================================
        # 4) PASS 2 — Build TRAIN/TEST windows using scaler_params
        # ============================================================
        X_train, y_train, src_train = [], [], []
        X_test, y_test = [], []

        self.timestamps_test = None
        self.ground_truth_test = None
        self.window_starts_test = []
        self.raw_test_acc = None
        self.test_sessions_raw = []

        test_offset = 0
        time_offset = 0.0
        gap = 1.0

        for s in sessions:
            X, y, window_starts = preprocess_acc_data(
                s["sensors"], s["ts"], s["bites"],
                window_size=self.window_size,
                stride=self.stride,
                scaler_params=self.scaler_params
            )

            if X.size == 0:
                continue

            if s["idx"] in test_idx:
                X_test.append(X)
                y_test.append(y)
                self.test_sessions_raw.append({
                    "idx": s["idx"],
                    "time_offset": time_offset,  # ⬅️ ΚΡΙΣΙΜΟ
                    "timestamps": s["ts"].copy(),  # native timeline
                    "sensors": s["sensors"].copy(),  # raw acc
                    "bites": s["bites"].copy()  # native GT
                })

                ws = np.asarray(window_starts, dtype=int)
                self.window_starts_test.extend((ws + test_offset).tolist())

                ts_shift = s["ts"] + time_offset
                bites_shift = s["bites"] + time_offset

                acc_shift = s["sensors"]  # raw acc, ΔΕΝ αλλάζει με offset (μόνο χρόνος αλλάζει)

                if self.raw_test_acc is None:
                    self.raw_test_acc = acc_shift
                else:
                    self.raw_test_acc = np.vstack([self.raw_test_acc, acc_shift])


            

                if self.timestamps_test is None:
                    self.timestamps_test = ts_shift
                    self.ground_truth_test = bites_shift
                else:
                    self.timestamps_test = np.concatenate([self.timestamps_test, ts_shift])
                    self.ground_truth_test = np.vstack([self.ground_truth_test, bites_shift])

                test_offset += len(s["ts"])
                time_offset = ts_shift[-1] + gap

            else:
                X_train.append(X)
                y_train.append(y)
                src_train.append(np.zeros((X.shape[0],), dtype=np.uint8))  # 0 = FIC

        # ============================================================
        # 5) FreeFIC negatives ONLY to TRAIN (use SAME scaler_params)
        # ============================================================
        free_meal_gt = freefic["meal_gt"]

        free_total_windows = 0
        free_total_pos = 0
        free_total_neg = 0
        free_used_sessions = 0

        for i, sess in enumerate(free_raw):
            acc_full = sess["acc"]
            ts_free = acc_full[:, 0]
            acc_free = acc_full[:, 1:4]
            meals = free_meal_gt[i]

            X_neg, y_neg = self.extract_non_meal_windows(acc_free, ts_free, meals)
            if X_neg.size > 0:
                free_used_sessions += 1
                free_total_windows += X_neg.shape[0]
                free_total_pos += int(np.sum(y_neg == 1))
                free_total_neg += int(np.sum(y_neg == 0))
                X_train.append(X_neg)
                y_train.append(y_neg)
                src_train.append(np.ones((X_neg.shape[0],), dtype=np.uint8))  # 1 = FreeFIC

        print("\n=== FreeFIC ADDITION CHECK ===")
        print("FreeFIC sessions used :", free_used_sessions, "/", len(free_raw))
        print("FreeFIC windows added :", free_total_windows)
        print("FreeFIC positives     :", free_total_pos, " (MUST be 0)")
        print("FreeFIC negatives     :", free_total_neg)
        print("==============================\n")
        # ============================================================
        # 6) Concatenate + compatibility
        # ============================================================
        if X_train:
            ch = X_train[0].shape[-1]
        elif X_test:
            ch = X_test[0].shape[-1]
        else:
            ch = 6

        self.X_train = np.concatenate(X_train, axis=0) if X_train else np.empty((0, self.window_size, ch), dtype=np.float32)
        self.y_train = np.concatenate(y_train, axis=0) if y_train else np.empty((0,), dtype=int)
        self.X_test = np.concatenate(X_test, axis=0) if X_test else np.empty((0, self.window_size, ch), dtype=np.float32)
        self.y_test = np.concatenate(y_test, axis=0) if y_test else np.empty((0,), dtype=int)
        self.src_train = np.concatenate(src_train, axis=0) if src_train else np.empty((0,), dtype=np.uint8)
        assert len(self.src_train) == len(self.y_train), "src_train not aligned with y_train"

        print("=== TRAIN FINAL SIZE CHECK ===")
        print("Total train windows:", self.X_train.shape[0])
        print("Total train pos    :", int(np.sum(self.y_train == 1)))
        print("Total train neg    :", int(np.sum(self.y_train == 0)))
        print("==============================")

        # Backward compatibility
        self.X, self.y = self.X_train, self.y_train

        # ============================================================
        # 7) Sanity checks (alignment)
        # ============================================================
        if self.X_test.shape[0] > 0:
            ws = np.asarray(self.window_starts_test, dtype=int)
            ts = self.timestamps_test
            gt = self.ground_truth_test

            assert ws.min() >= 0, f"window_starts_test has negative index: {ws.min()}"
            assert ws.max() + (self.window_size - 1) < len(ts), \
                f"window end out of bounds: max_start={ws.max()}, len(ts)={len(ts)}"

            end_times = ts[ws + (self.window_size - 1)]
            assert np.all(np.diff(end_times) >= 0), "Window end times are not monotonic!"

            if gt is not None and len(gt) > 0:
                t0, t1 = ts[0], ts[-1]
                gt_min, gt_max = float(np.min(gt)), float(np.max(gt))
                assert gt_min >= t0 - 1e-6 and gt_max <= t1 + 1e-6, \
                    f"GT outside timeline: gt=[{gt_min},{gt_max}] timeline=[{t0},{t1}]"

            print("✅ Sanity checks passed: test alignment looks correct")

        if self.raw_test_acc is not None:
            print("raw_test_acc shape:", self.raw_test_acc.shape)
            print("timestamps_test len:", len(self.timestamps_test))

        print(f"✅ setup(full) -> Train: {self.X_train.shape}, Test: {self.X_test.shape}")

        # Optional distribution print
        print("=== CLASS DISTRIBUTION CHECK ===")
        print("Train windows   :", len(self.y_train))
        print("Train positives :", int(np.sum(self.y_train)))
        print("Train negatives :", int(len(self.y_train) - np.sum(self.y_train)))
        print("Train pos rate  :", float(np.mean(self.y_train)) if len(self.y_train) else 0.0)
        print("Test windows    :", len(self.y_test))
        print("Test positives  :", int(np.sum(self.y_test)))
        print("Test pos rate   :", float(np.mean(self.y_test)) if len(self.y_test) else 0.0)
        print("================================")

    # ============================================================
    # LOSO SETUP (optional) — computes scaler on TRAIN subjects only
    # ============================================================
    def setup_LOSO(self, test_subject_id):
        with open(self.pkl_path, "rb") as fh:
            dataset = pkl.load(fh)

        raw_data = dataset["signals_raw"]
        bite_gt = dataset["bite_gt"]
        subject_ids = dataset["subject_id"]

        # collect eligible sessions indices by subject
        eligible = []
        for i, session in enumerate(raw_data):
            acc_full = session[self.sensor_type]
            sensors = acc_full[:, 1:4]
            if len(sensors) < self.window_size:
                continue
            if (len(sensors) - self.window_size) < 0:
                continue
            eligible.append(i)

        # TRAIN idx = eligible excluding test_subject
        train_idx = []
        test_idx = []
        for i in eligible:
            sid = int(subject_ids[i])
            if sid == int(test_subject_id):
                test_idx.append(i)
            else:
                train_idx.append(i)

        if len(test_idx) == 0:
            raise ValueError(f"No eligible sessions found for test_subject_id={test_subject_id}")

        # PASS 1: scaler on TRAIN only
        print(f"🔄 LOSO: Calculating Global Scaler Stats on TRAIN only (test subj={test_subject_id})...")
        scaler_low = StandardScaler()
        scaler_high = StandardScaler()

        for i in train_idx:
            acc_full = raw_data[i][self.sensor_type]
            sensors = acc_full[:, 1:4]
            low, high = apply_filters(sensors)
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        # include FreeFIC as well (optional but recommended)
        with open(self.freefic_path, "rb") as fh:
            freefic = pkl.load(fh)
        free_raw = freefic["signals_raw"]

        for sess in free_raw:
            acc_full = sess["acc"]
            acc_free = acc_full[:, 1:4]
            low, high = apply_filters(acc_free)
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        self.scaler_params = {
            "mean_low": scaler_low.mean_.astype(np.float32),
            "scale_low": scaler_low.scale_.astype(np.float32),
            "mean_high": scaler_high.mean_.astype(np.float32),
            "scale_high": scaler_high.scale_.astype(np.float32),
        }
        print("✅ LOSO scaler stats ready.")

        # PASS 2: build windows
        X_train, y_train, X_test, y_test = [], [], [], []
        self.timestamps_test = None
        self.ground_truth_test = None
        self.window_starts_test = []

        test_offset = 0
        time_offset = 0.0
        gap = 1.0

        for i in train_idx:
            acc_full = raw_data[i][self.sensor_type]
            ts = acc_full[:, 0]
            sensors = acc_full[:, 1:4]
            bites = bite_gt[i]

            X, y, _ = preprocess_acc_data(
                sensors, ts, bites,
                window_size=self.window_size,
                stride=self.stride,
                scaler_params=self.scaler_params
            )
            if X.size:
                X_train.append(X)
                y_train.append(y)

        for i in test_idx:
            acc_full = raw_data[i][self.sensor_type]
            ts = acc_full[:, 0]
            sensors = acc_full[:, 1:4]
            bites = bite_gt[i]

            X, y, window_starts = preprocess_acc_data(
                sensors, ts, bites,
                window_size=self.window_size,
                stride=self.stride,
                scaler_params=self.scaler_params
            )
            if X.size == 0:
                continue

            X_test.append(X)
            y_test.append(y)

            ws = np.asarray(window_starts, dtype=int)
            self.window_starts_test.extend((ws + test_offset).tolist())

            ts_shift = ts + time_offset
            bites_shift = bites + time_offset

            if self.timestamps_test is None:
                self.timestamps_test = ts_shift
                self.ground_truth_test = bites_shift
            else:
                self.timestamps_test = np.concatenate([self.timestamps_test, ts_shift])
                self.ground_truth_test = np.vstack([self.ground_truth_test, bites_shift])

            test_offset += len(ts)
            time_offset = ts_shift[-1] + gap

        # FreeFIC negatives ONLY to TRAIN
        free_meal_gt = freefic["meal_gt"]
        for i, sess in enumerate(free_raw):
            acc_full = sess["acc"]
            ts_free = acc_full[:, 0]
            acc_free = acc_full[:, 1:4]
            meals = free_meal_gt[i]
            X_neg, y_neg = self.extract_non_meal_windows(acc_free, ts_free, meals)
            if X_neg.size:
                X_train.append(X_neg)
                y_train.append(y_neg)

        self.X_train = np.concatenate(X_train, axis=0) if X_train else np.empty((0, self.window_size, 6), dtype=np.float32)
        self.y_train = np.concatenate(y_train, axis=0) if y_train else np.empty((0,), dtype=int)
        self.X_test = np.concatenate(X_test, axis=0) if X_test else np.empty((0, self.window_size, 6), dtype=np.float32)
        self.y_test = np.concatenate(y_test, axis=0) if y_test else np.empty((0,), dtype=int)

        self.X, self.y = self.X_train, self.y_train

        print(f"✅ LOSO Fold - Subject {test_subject_id}")
        print(f"   Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    # ============================================================
    # DATALOADERS
    # ============================================================
    def get_dataloader(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        return dataset.shuffle(1000).batch(self.batch_size)

    def get_dataloader_LOSO(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        return dataset.shuffle(1000).batch(self.batch_size)

    def get_balanced_dataloader_2(self):

        # Βρίσκουμε τα indices για τις 3 κατηγορίες


        pos_idx = np.where(self.y_train == 1)[0]   # 1. Θετικά (Μπουκιές)

        # ΠΡΟΣΟΧΗ: Αυτό προϋποθέτει ότι έχεις το self.src_train φορτωμένο!
        neg_fic_idx = np.where((self.y_train == 0) & (self.src_train == 0))[0]  # 2. Αρνητικά από FIC
        neg_free_idx = np.where((self.y_train == 0) & (self.src_train == 1))[0] # 3. Αρνητικά από FreeFIC (src=1)

        print(f"Stats: POS={len(pos_idx)}, NEG_FIC={len(neg_fic_idx)}, NEG_FREE={len(neg_free_idx)}")

        # Φτιάχνουμε τα 3 Datasets (Χωρίς βοηθητική συνάρτηση)

        # Dataset Μπουκιών
        pos_ds = tf.data.Dataset.from_tensor_slices(
            (self.X_train[pos_idx], self.y_train[pos_idx])
        ).repeat()  # Το repeat() βοηθάει να μην τελειώσουν γρήγορα τα δεδομένα κατά το sampling

        # Dataset FIC (Ηρεμία)
        neg_fic_ds = tf.data.Dataset.from_tensor_slices(
            (self.X_train[neg_fic_idx], self.y_train[neg_fic_idx])
        ).repeat()

        # Dataset FreeFIC (Κινήσεις)
        neg_free_ds = tf.data.Dataset.from_tensor_slices(
            (self.X_train[neg_free_idx], self.y_train[neg_free_idx])
        ).repeat()

        ds = tf.data.Dataset.sample_from_datasets(
            [pos_ds, neg_fic_ds, neg_free_ds],
            weights=[0.3, 0.5, 0.2]
        )

        ds = (ds
              .shuffle(buffer_size=10000)
              .batch(self.batch_size)
              .prefetch(tf.data.AUTOTUNE))

        return ds
    def get_balanced_dataloader(self):
        X = getattr(self, "X_train", self.X)
        y = getattr(self, "y_train", self.y)

        X_pos = X[y == 1]
        X_neg = X[y == 0]

        if len(X_pos) == 0:
            print("Warning: No positive samples in training set!")
            return tf.data.Dataset.from_tensor_slices((X_neg, np.zeros(len(X_neg), dtype=int))).batch(self.batch_size)

        pos_ds = tf.data.Dataset.from_tensor_slices((X_pos, np.ones(len(X_pos), dtype=int))).shuffle(1000).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices((X_neg, np.zeros(len(X_neg), dtype=int))).shuffle(1000).repeat()
        neg_idx = np.where(self.y_train == 0)[0]
        # εδώ φτιαχνω 2 ροες δεδομένων μια 0 και μια 1 και κάτω λέω να παίρνει 50-50
        print("NEG FreeFIC ratio:", self.src_train[neg_idx].mean())
        ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def get_balanced_dataloader_LOSO(self):
        X_pos = self.X_train[self.y_train == 1]
        X_neg = self.X_train[self.y_train == 0]

        pos_ds = tf.data.Dataset.from_tensor_slices((X_pos, np.ones(len(X_pos), dtype=int))).shuffle(1000).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices((X_neg, np.zeros(len(X_neg), dtype=int))).shuffle(1000).repeat()
        ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    # ============================================================
    # FreeFIC helper (negatives outside meals) — uses SAME scaler
    # ============================================================
    #
    def extract_non_meal_windows(self, acc_data, timestamps, meal_intervals):
        if self.scaler_params is None:
            raise RuntimeError("scaler_params not set. Run setup() first.")

        acc_data = np.asarray(acc_data)
        timestamps = np.asarray(timestamps)

        # 1) Build ALL windows on the original (uncut) signal
        dummy_bites = np.zeros((0, 2), dtype=np.float32)

        X_all, y_all, window_starts = preprocess_acc_data(
            acc_data, timestamps, dummy_bites,
            window_size=self.window_size,
            stride=self.stride,
            scaler_params=self.scaler_params
        )
        # y_all should be all zeros already (dummy bites)

        if X_all.size == 0:
            return X_all, y_all

        ws = np.asarray(window_starts, dtype=int)
        t_end = timestamps[ws + self.window_size - 1]  # window_end times

        # 2) Keep only windows whose window_end is OUTSIDE all meal intervals
        keep = np.ones_like(t_end, dtype=bool)
        for start, end in meal_intervals:
            keep &= ~((t_end >= start) & (t_end <= end))

        X = X_all[keep]
        y = y_all[keep]  # all zeros
        return X, y


if __name__ == "__main__":
    dm = WindowsDataModule()
    dm.setup()
