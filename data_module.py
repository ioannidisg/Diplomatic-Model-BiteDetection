import pickle as pkl
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import apply_filters


class WindowsDataModule:
    def __init__(self):
        self.pkl_path = "data/FIC.pkl"
        self.freefic_path = "data/FreeFIC_FreeFIC-heldout.pkl"

        self.batch_size = 64
        self.window_size = 500

        # ✅ LOCKED STRIDES (per your request)
        # fs = 100 Hz  ->  stride=5  => 0.05s  (FIC)
        # fs = 100 Hz  ->  stride=100=> 1.0s   (FreeFIC)
        self.stride_fic = 5
        self.stride_freefic = 100

        # keep for backward-compatibility with older code (treat as FIC stride)
        self.stride = self.stride_fic

        self.sensor_type = "acc"
        self.fs = 100.0

        self.scaler_params = None
        self.train_sessions = []
        self.test_sessions = []

        # (kept, in case other scripts use them later)
        self.timestamps_test = None
        self.ground_truth_test = None
        self.window_starts_test = []


    def setup(self, stage=None, split="global_by_session", test_ratio=0.2, seed=42):
        rng = np.random.RandomState(seed)

        # =========================
        # FILTER CONSTANTS (for consistent filters + ts_eff delay)
        # =========================
        fs = self.fs
        l_hp = 513
        delay_samples = (l_hp - 1) // 2
        delay_sec = delay_samples / fs

        # 1) Load FIC
        with open(self.pkl_path, "rb") as fh:
            dataset = pkl.load(fh)

        raw_data = dataset["signals_raw"]
        bite_gt = dataset["bite_gt"]
        subject_ids = dataset.get("subject_id", None)

        # 2) Collect Eligible Sessions (metadata only)
        sessions = []
        for i, session in enumerate(raw_data):
            acc_full = session[self.sensor_type]
            timestamps = acc_full[:, 0]
            sensors = acc_full[:, 1:4]
            bites = bite_gt[i]

            sessions.append({
                "idx": i,
                "subj": int(subject_ids[i]) if subject_ids is not None else None,
                "ts": timestamps,
                "sensors": sensors,
                "bites": bites
            })

        # 3) Split Train/Test indices
        if split == "per_subject" and subject_ids is not None:
            subj2idx = defaultdict(list)
            for s in sessions:
                subj2idx[s["subj"]].append(s["idx"])

            test_idx = set()
            for sid, idxs in subj2idx.items():
                idxs = list(idxs)
                rng.shuffle(idxs)
                n_test = max(1, int(np.ceil(len(idxs) * test_ratio)))
                test_idx.update(idxs[:n_test])
        else:
            idxs = [s["idx"] for s in sessions]
            rng.shuffle(idxs)
            n_test = max(1, int(np.ceil(len(idxs) * test_ratio)))
            test_idx = set(idxs[:n_test])

        # 4) Scaler stats (Train-only FIC) + (Optional) FreeFIC (sid<=14)
        print("🔄 Calculating Scaler Stats...")
        scaler_low = StandardScaler()
        scaler_high = StandardScaler()

        # --- FIC train only ---
        for s in sessions:
            if s["idx"] in test_idx:
                continue
            if len(s["sensors"]) < self.window_size:
                continue

            low, high = apply_filters(s["sensors"], fs=fs, l_hp=l_hp)
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        # --- Load FreeFIC once (used also below) ---
        with open(self.freefic_path, "rb") as fh:
            freefic = pkl.load(fh)

        free_raw = freefic["signals_raw"]
        free_subject_ids = freefic.get("subject_id", None)

        # FreeFIC scaling: include only training subjects (sid<=14), skip held-out (sid>=15)
        for i, sess in enumerate(free_raw):
            sid = int(free_subject_ids[i]) if free_subject_ids is not None else None
            if sid is not None and sid >= 15:
                continue

            acc_free = sess["acc"][:, 1:4]
            if len(acc_free) < self.window_size:
                continue

            low, high = apply_filters(acc_free, fs=fs, l_hp=l_hp)
            scaler_low.partial_fit(low)
            scaler_high.partial_fit(high)

        self.scaler_params = {
            "mean_low": scaler_low.mean_.astype(np.float32),
            "scale_low": scaler_low.scale_.astype(np.float32),
            "mean_high": scaler_high.mean_.astype(np.float32),
            "scale_high": scaler_high.scale_.astype(np.float32),
        }
        print("✅ Scaler ready.")

        # 5) Build TRAIN (Lazy) & TEST (Lazy) session dicts
        self.train_sessions = []
        self.test_sessions = []

        time_offset, test_offset, gap = 0.0, 0, 1.0

        print("🔨 Processing Sessions (Applying Filters & Lazy Setup)...")

        # =========================
        # FIC LOOP
        # =========================
        for i, session in enumerate(raw_data):
            acc_full = session[self.sensor_type]
            ts = acc_full[:, 0]
            sensors = acc_full[:, 1:4]
            bites = bite_gt[i]

            if len(sensors) < self.window_size:
                continue

            low, high = apply_filters(sensors, fs=fs, l_hp=l_hp)
            ts_eff = ts - np.float32(delay_sec)

            s_dict = {
                "idx": i,
                "ts": ts,
                "ts_eff": ts_eff,
                "signal_raw": sensors.astype(np.float32),  # ⬅️ ΠΡΟΣΘΗΚΗ
                "signal_low": low,
                "signal_high": high,
                "events": bites,          # FIC: BITES (end timestamps used later)
                "type": "fic",
                "time_offset": 0.0,
                "test_offset": 0
            }

            if i in test_idx:
                # === FIC TEST ===
                s_dict["time_offset"] = time_offset
                s_dict["test_offset"] = test_offset

                # shifted events ONLY for FIC (bites)
                if len(bites) > 0:
                    s_dict["events_shifted"] = (np.asarray(bites, dtype=np.float32) - np.float32(
                        delay_sec)) + np.float32(time_offset)
                else:
                    s_dict["events_shifted"] = np.empty((0, 2))

                self.test_sessions.append(s_dict)

                duration = ts[-1]
                time_offset += duration + gap
                test_offset += len(ts)
            else:
                # === FIC TRAIN ===
                self.train_sessions.append(s_dict)

        # =========================
        # FreeFIC LOOP (Negatives to Train, Held-out to Test)
        # =========================
        free_meal_gt = freefic["meal_gt"]
        free_ids = freefic.get("subject_id", None)

        for i, sess in enumerate(free_raw):
            sid = int(free_ids[i]) if free_ids is not None else None

            acc_full = sess["acc"]
            ts = acc_full[:, 0]
            sensors = acc_full[:, 1:4]

            if len(sensors) < self.window_size:
                continue

            low, high = apply_filters(sensors, fs=fs, l_hp=l_hp)
            ts_eff = ts - np.float32(delay_sec)

            meals = free_meal_gt[i]  # FreeFIC: MEALS (intervals)

            s_dict = {
                "idx": f"free_{i}",
                "ts": ts,
                "ts_eff": ts_eff,
                "signal_raw": sensors.astype(np.float32),  # ⬅️ ΠΡΟΣΘΗΚΗ
                "signal_low": low,
                "signal_high": high,
                "events": meals,          # FreeFIC: MEALS intervals
                "type": "freefic",
                "time_offset": 0.0,
                "test_offset": 0
                # NO events_shifted here (meals are not bites)
            }

            if sid is not None and sid <= 14:
                # === FreeFIC TRAIN ===
                self.train_sessions.append(s_dict)

            elif sid is not None and sid >= 15:
                # === FreeFIC TEST ===
                s_dict["time_offset"] = time_offset
                s_dict["test_offset"] = test_offset

                self.test_sessions.append(s_dict)

                duration = ts[-1]
                time_offset += duration + gap
                test_offset += len(ts)

        print(f"✅ Setup Done. Train Sessions (Lazy): {len(self.train_sessions)}")
        print(f"✅ Setup Done. Test Sessions  (Lazy): {len(self.test_sessions)}")
        print(f"🔒 Strides locked: FIC={self.stride_fic} | FreeFIC={self.stride_freefic} | fs={self.fs}Hz")


if __name__ == "__main__":
    dm = WindowsDataModule()
    dm.setup()
