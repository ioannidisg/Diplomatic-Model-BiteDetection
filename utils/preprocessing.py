import numpy as np
from scipy.signal import firwin, lfilter
from sklearn.preprocessing import StandardScaler


def apply_filters(acc_data, l_ma=25, f_c=1.0, l_hp=513, fs=100):

    data = np.asarray(acc_data, dtype=np.float32).copy()

    # Axis invert 
#    data[:, 0] *= -1.0

    # Moving average smoothing (same length)
    kernel = np.ones(l_ma, dtype=np.float32) / np.float32(l_ma)
    smooth = np.zeros_like(data, dtype=np.float32)
    for i in range(3):
        smooth[:, i] = np.convolve(data[:, i], kernel, mode="same").astype(np.float32)

    # FIR filters
    b_low = firwin(l_hp, f_c, pass_zero=True, fs=fs).astype(np.float32)
    b_high = firwin(l_hp, f_c, pass_zero=False, fs=fs).astype(np.float32)

    low_filtered = np.zeros_like(smooth, dtype=np.float32)
    high_filtered = np.zeros_like(smooth, dtype=np.float32)

    for i in range(3):
        low_filtered[:, i] = lfilter(b_low, 1.0, smooth[:, i]).astype(np.float32)
        high_filtered[:, i] = lfilter(b_high, 1.0, smooth[:, i]).astype(np.float32)

    return low_filtered, high_filtered



def extract_y_old(timestamps_eff, window_size, bite_intervals, window_starts,
              epsilon=0.2):
    """
    positive if window END is within ±epsilon of a BITE END.
    timestamps_eff: timestamps after delay correction.
    """
    n_windows = len(window_starts)
    y = np.zeros(n_windows, dtype=int)

    if bite_intervals is None or np.size(bite_intervals) == 0 or n_windows == 0:
        return y

    bite_intervals = np.asarray(bite_intervals)
    bite_ends = bite_intervals[:, 1]

    for k, start in enumerate(window_starts):
        t_end = timestamps_eff[start + window_size - 1]
        if np.any(np.abs(t_end - bite_ends) <= epsilon):
            y[k] = 1

    return y

def extract_y( timestamps_eff,window_size,bite_intervals,window_starts,mode="fic",epsilon=0.2):
    """
      - "fic"      : intervals = bite intervals → y ∈ {0,1}
      - "freefic"  : intervals = meal intervals → y ∈ {0,-1}
    """

    y = []

    for s in window_starts:
        t_end = timestamps_eff[s + window_size - 1]

        if mode == "freefic":
            # μέσα σε meal → N/A
            inside_meal = np.any(
                (t_end >= bite_intervals[:, 0]) &
                (t_end <= bite_intervals[:, 1])
            )
            if inside_meal:
                y.append(-1)   # N/A
            else:
                y.append(0)    # σίγουρα non-meal
            continue

        # mode == "fic"
        if bite_intervals.size == 0:
            y.append(0)
        else:
            bite_ends = bite_intervals[:, 1]
            is_pos = np.any(np.abs(t_end - bite_ends) <= epsilon)
            y.append(1 if is_pos else 0)

    return np.asarray(y, dtype=int)



def preprocess_acc_data(acc_data, timestamps, bite_intervals,
                        window_size=500, stride=20,
                        scaler_params=None,
                        epsilon=0.625,
                        l_ma=25, f_c=1.0, l_hp=513, fs=100,label_mode = 'fic'):


    acc_data = np.asarray(acc_data, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32)

    # 1) Filters
    low_filtered, high_filtered = apply_filters(
        acc_data, l_ma=l_ma, f_c=f_c, l_hp=l_hp, fs=fs
    )

    # 2) Scaling
    if scaler_params is not None:
        mean_low = np.asarray(scaler_params["mean_low"], dtype=np.float32)
        scale_low = np.asarray(scaler_params["scale_low"], dtype=np.float32)
        mean_high = np.asarray(scaler_params["mean_high"], dtype=np.float32)
        scale_high = np.asarray(scaler_params["scale_high"], dtype=np.float32)

        acc_low = (low_filtered - mean_low) / (scale_low + 1e-8)
        acc_high = (high_filtered - mean_high) / (scale_high + 1e-8)
    else:
        # fallback local scaling (όχι για τελικό, αλλά για safety)
        s_low = StandardScaler()
        s_high = StandardScaler()
        acc_low = s_low.fit_transform(low_filtered).astype(np.float32)
        acc_high = s_high.fit_transform(high_filtered).astype(np.float32)

    acc_scaled = np.concatenate([acc_low, acc_high], axis=1).astype(np.float32)  # (N,6)

    # 3) Windowing
    X = []
    window_starts = []
    n = len(acc_scaled)
    for start in range(0, n - window_size + 1, stride):
        X.append(acc_scaled[start:start + window_size])
        window_starts.append(start)

    if len(X) == 0:
        return np.empty((0, window_size, 6), dtype=np.float32), np.empty((0,), dtype=int), []

    X = np.stack(X, axis=0).astype(np.float32)

    # 4) Delay correction
    delay_samples = (l_hp - 1) // 2
    timestamps_eff = timestamps - np.float32(delay_samples / fs)

    # 5) Labels
    y = extract_y(
        timestamps_eff=timestamps_eff,
        window_size=window_size,
        bite_intervals=bite_intervals,
        window_starts=window_starts,
        epsilon=epsilon,
        mode=label_mode
    ).astype(int)

    return X, y, window_starts
