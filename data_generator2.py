# data_generator.py
import numpy as np
import tensorflow as tf
from utils.preprocessing import apply_filters


class BalancedWindowSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        sessions,
        batch_size,
        window_size,
        stride,                 # FIC stride
        scaler_params,
        epsilon,
        ratio=1,
        stride_freefic=None,    # FreeFIC stride 
        steps_per_epoch=None,   
        seed=42

    ):
        self.sessions = sessions
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.ratio = int(ratio)
        self.epsilon = float(epsilon)

        self.stride_fic = int(stride)
        self.stride_freefic = int(stride_freefic) if stride_freefic is not None else int(stride)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else None

        # RNG for reproducible shuffles/sampling
        self.rng = np.random.RandomState(seed)

        # Cache scaler arrays (broadcast-friendly)
        self.mean_low = np.asarray(scaler_params["mean_low"], dtype=np.float32)
        self.scale_low = np.asarray(scaler_params["scale_low"], dtype=np.float32)
        self.mean_high = np.asarray(scaler_params["mean_high"], dtype=np.float32)
        self.scale_high = np.asarray(scaler_params["scale_high"], dtype=np.float32)

        self.pos_indices = []
        self.neg_indices = []

        print(
            f"Generator Init: FIC stride={self.stride_fic}, FreeFIC stride={self.stride_freefic}, "
            f"Ratio=1:{self.ratio}, steps_per_epoch={self.steps_per_epoch}"
        )

        # SCANNING 
        for sess_idx, s in enumerate(self.sessions):
            if "signal_low" not in s:
                continue

            s_type = s.get("type", "fic")
            stride_used = self.stride_freefic if s_type == "freefic" else self.stride_fic

            n_samples = len(s["signal_low"])
            if n_samples < self.window_size:
                continue

            window_starts = np.arange(0, n_samples - self.window_size + 1, stride_used, dtype=np.int64)

            events = s.get("events", None)
            delay = s["ts"][0] - s["ts_eff"][0]  # FIR group delay in seconds
            if events is None or len(events) == 0:
                # No events: everything is NEG
                for w in window_starts:
                    self.neg_indices.append((sess_idx, int(w), 0))
                continue

            ts_eff = s["ts_eff"]
            t_ends = ts_eff[window_starts + (self.window_size - 1)]

            if s_type == "freefic":
                # FreeFIC: keep only NEG 
                ev_starts = events[:, 0] - delay
                ev_ends = events[:, 1] - delay

                for i, t_end in enumerate(t_ends):
                    inside_any = np.any((t_end >= ev_starts) & (t_end <= ev_ends))
                    if not inside_any:
                        self.neg_indices.append((sess_idx, int(window_starts[i]), 0))
            else:
                # FIC: POS if window end is within epsilon of any bite end timestamp
                event_ends = events[:, 1] - delay
                for i, t_end in enumerate(t_ends):
                    w_start = int(window_starts[i])
                    is_pos = np.any(np.abs(t_end - event_ends) <= self.epsilon)
                    if is_pos:
                        self.pos_indices.append((sess_idx, w_start, 1))
                    else:
                        self.neg_indices.append((sess_idx, w_start, 0))

        self.pos_indices = np.asarray(self.pos_indices, dtype=object)
        self.neg_indices = np.asarray(self.neg_indices, dtype=object)

        print(f"Generator Ready: {len(self.pos_indices)} POS, {len(self.neg_indices)} NEG.")

        if len(self.pos_indices) == 0:
            raise RuntimeError("No POS windows found. Check epsilon/labels or session types.")
        if len(self.neg_indices) == 0:
            raise RuntimeError("No NEG windows found. Check FreeFIC intervals/scanning.")

        # POS permutation for (mostly) non-replacement sampling within an epoch
        self._reset_pos_perm()

    # Keras Sequence API
    def __len__(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch

        # Default: epoch long enough to (approximately) cover all positives once
        pos_per_batch = max(1, self.batch_size // (self.ratio + 1))
        return int(np.ceil(len(self.pos_indices) / pos_per_batch))

    def on_epoch_end(self):
        self._reset_pos_perm()

    def __getitem__(self, idx):
        n_pos = max(1, self.batch_size // (self.ratio + 1))
        n_neg = self.batch_size - n_pos

        # POS: take a slice from a shuffled permutation (wrap if needed)
        pos_sel = self._take_pos(n_pos)

        # NEG: random sampling
        neg_sel_idx = self.rng.randint(0, len(self.neg_indices), size=n_neg)
        neg_sel = self.neg_indices[neg_sel_idx]

        batch_items = np.concatenate([pos_sel, neg_sel], axis=0)
        self.rng.shuffle(batch_items)

        X_batch = np.empty((self.batch_size, self.window_size, 6), dtype=np.float32)
        y_batch = np.empty((self.batch_size,), dtype=np.int32)

        for j, (sess_idx, w_start, label) in enumerate(batch_items):
            sess_idx = int(sess_idx)
            w_start = int(w_start)
            w_end = w_start + self.window_size

            s = self.sessions[sess_idx]

            low_seq = s["signal_low"][w_start:w_end].astype(np.float32)  # (T,3)
            high_seq = s["signal_high"][w_start:w_end].astype(np.float32)  # (T,3)

            low_seq, Q = self.rotate_raw_acc(low_seq)
            high_seq = (high_seq @ Q.T).astype(np.float32)

            low_norm = (low_seq - self.mean_low) / (self.scale_low + 1e-8)
            high_norm = (high_seq - self.mean_high) / (self.scale_high + 1e-8)

            X_batch[j] = np.hstack([low_norm, high_norm]).astype(np.float32)
            y_batch[j] = int(label)

        return X_batch, y_batch

    #  helpers
    def _reset_pos_perm(self):
        self.pos_perm = np.arange(len(self.pos_indices), dtype=np.int64)
        self.rng.shuffle(self.pos_perm)
        self.pos_ptr = 0

    def _take_pos(self, n_pos):
        # take n_pos indices from perm, wrapping if needed
        if self.pos_ptr + n_pos <= len(self.pos_perm):
            sel = self.pos_perm[self.pos_ptr:self.pos_ptr + n_pos]
            self.pos_ptr += n_pos
            return self.pos_indices[sel]

        # wrap around
        tail = self.pos_perm[self.pos_ptr:]
        need = n_pos - len(tail)
        self._reset_pos_perm()
        head = self.pos_perm[:need]
        sel = np.concatenate([tail, head], axis=0)
        self.pos_ptr = need
        return self.pos_indices[sel]

    def rotate_raw_acc(self, acc):
        """
        Random small rotation augmentation.
        Returns:
          acc_rot: (T,3)
          Q:      (3,3) rotation matrix (identity if no rotation)
        """
        acc = np.asarray(acc, dtype=np.float32)

        # ~50%: no rotation
        if self.rng.rand() > 0.5:
            return acc, np.eye(3, dtype=np.float32)

        theta_x = (self.rng.normal(0, 10) * np.pi / 180.0).astype(np.float32)
        theta_z = (self.rng.normal(0, 10) * np.pi / 180.0).astype(np.float32)

        Qx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ], dtype=np.float32)

        Qz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        mode = self.rng.randint(0, 4)
        if mode == 0:
            Q = Qx
        elif mode == 1:
            Q = Qz
        elif mode == 2:
            Q = Qx @ Qz
        else:
            Q = Qz @ Qx

        acc_rot = (acc @ Q.T).astype(np.float32)
        return acc_rot, Q

