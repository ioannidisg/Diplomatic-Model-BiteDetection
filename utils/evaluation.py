import numpy as np
from scipy.signal import find_peaks


import numpy as np
from scipy.signal import find_peaks

import numpy as np

def extract_detected_times_paper_localmax_2(
    probs,
    window_starts,
    timestamps,
    window_size,
    threshold=0.89,
    min_distance_sec=2.0,
    stride=5,
    fs=100.0,
    K=1,
):
    # Align lengths
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    ws = np.asarray(window_starts, dtype=np.int64)
    n = min(len(p), len(ws))
    p = p[:n]
    ws = ws[:n]

    # t_end per window (end timestamp of each window)
    ts = np.asarray(timestamps, dtype=np.float64)
    end_idx = ws + (window_size - 1)
    valid = (end_idx >= 0) & (end_idx < len(ts))

    p = p[valid]
    end_idx = end_idx[valid]
    t_end = ts[end_idx]

    # 1) thresholding exactly like paper: replace with zeros
    p_thr = p.copy()
    p_thr[p_thr < threshold] = 0.0

    if K > 1:
        m = (p_thr > 0.0)  # boolean mask above threshold
        keep = np.zeros_like(m, dtype=bool)

        # find runs of True in m
        i = 0
        L = len(m)
        while i < L:
            if not m[i]:
                i += 1
                continue
            j = i
            while j < L and m[j]:
                j += 1
            # run is [i, j)
            if (j - i) >= K:
                keep[i:j] = True
            i = j

        p_thr = p_thr * keep.astype(np.float64)

    # 2local maxima with min distance 2 sec
    frame_rate = fs / stride  # windows per second
    distance = max(1, int(np.ceil(min_distance_sec * frame_rate)))
    peaks, _ = find_peaks(p_thr, distance=distance)

    # output: bite timestamps
    bite_times = t_end[peaks]
    return bite_times

def extract_detected_times_paper_localmax(
    probs,
    window_starts,
    timestamps,
    window_size,
    threshold=0.89,
    min_distance_sec=2.0,
    stride=5,
    fs=100.0
):
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    ws = np.asarray(window_starts, dtype=np.int64)
    n = min(len(p), len(ws))
    p = p[:n]
    ws = ws[:n]

    # t_end per window (end timestamp of each window)
    ts = np.asarray(timestamps, dtype=np.float64)
    end_idx = ws + (window_size - 1)
    valid = (end_idx >= 0) & (end_idx < len(ts))

    p = p[valid]
    end_idx = end_idx[valid]
    t_end = ts[end_idx]
    
    #  thresholding  like paper
    p_thr = p.copy()
    p_thr[p_thr < threshold] = 0.0

    #  local maxima with min distance 2 sec
    frame_rate = fs / stride  # windows per second
    distance = max(1, int(np.ceil(min_distance_sec * frame_rate)))
    peaks, _ = find_peaks(p_thr, distance=distance)

    bite_times = t_end[peaks]
    return bite_times


def extract_detected_times(y_pred_bin, window_starts, timestamps, window_size):
    detected_times = []
    for i, pred in enumerate(y_pred_bin):
        if pred == 1:
            # if i >= len(window_starts):  # Μου εβγαζε Index error αλλά τλκ διορθώθηκε όχι απο αυτό
            #     continue
            start_idx = window_starts[i]
            end_idx = start_idx + window_size - 1
            # if end_idx >= len(timestamps):
            #     continue
            t_start = timestamps[start_idx]
            t_end = timestamps[end_idx]
            detected_times.append((t_start, t_end))
    return detected_times


def calculate_f1_custom(detected_intervals, ground_truth_intervals, epsilon=0.2):
    TP = 0
    FP= 0
    matched_gt = set() # ειναι ένα κενό σύνολο

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

    print(f"TP: {TP} | FP: {FP} | FN: {FN} | Precision: {precision:.2f} | Recall: {recall:.2f}")

    return f1



import numpy as np
from scipy.signal import find_peaks
from scipy.signal import find_peaks
import numpy as np

def extract_bites_paper_style(probs, t_end, threshold=0.89, min_distance_sec=2.0, stride=5, fs=100.0):
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    ws = np.asarray(window_starts, dtype=np.int64)
    n = min(len(p), len(ws))
    p = p[:n]
    ws = ws[:n]

    # t_end per window (end timestamp of each window)
    ts = np.asarray(timestamps, dtype=np.float64)
    end_idx = ws + (window_size - 1)
    valid = (end_idx >= 0) & (end_idx < len(ts))

    p = p[valid]
    end_idx = end_idx[valid]
    t_end = ts[end_idx]

    # 1) thresholding exactly like paper: replace with zeros
    p_thr = p.copy()
    p_thr[p_thr < threshold] = 0.0

    # 2) local maxima with min distance 2 sec
    frame_rate = fs / stride  # windows per second
    distance = max(1, int(np.ceil(min_distance_sec * frame_rate)))
    peaks, _ = find_peaks(p_thr, distance=distance)

    # 3) output: bite timestamps (or convert to intervals if your code expects intervals)
    bite_times = t_end[peaks]
    return bite_times

def extract_detected_times_localmax(
    probs,
    window_starts,
    timestamps,
    window_size,
    threshold=0.9,
    min_distance_sec=5.0,
    stride=125,
    fs=100,
    debug=False
):
  
    probs = np.asarray(probs).reshape(-1)
    ws = np.asarray(window_starts, dtype=int)

    n = min(len(probs), len(ws))
    probs = probs[:n]
    ws = ws[:n]

    frame_rate = fs / stride              # windows per second
    min_distance_frames = int(min_distance_sec * frame_rate)

    peaks, props = find_peaks(
        probs,
        height=threshold,
        distance=max(1, min_distance_frames)
    )

    if debug:
        print(f"[DEBUG] peaks found: {len(peaks)} (threshold={threshold}, min_dist={min_distance_sec}s)")

    # --- build detected intervals ---
    detected_intervals = []
    ts = np.asarray(timestamps)

    for p in peaks:
        start_idx = ws[p]
        end_idx = start_idx + window_size - 1

        if start_idx < 0 or end_idx >= len(ts):
            continue

        detected_intervals.append(
            (float(ts[start_idx]), float(ts[end_idx]))
        )

    return detected_intervals

def extract_detected_times_clustermax(
    probs,
    window_starts,
    timestamps,
    window_size,
    threshold=0.89,
    min_distance_sec=5.0,
    stride=250,
    fs=100,
    K=2,
    debug=False
):
    """
    Build detected intervals from per-window probabilities using:
      1) thresholding (p >= threshold)
      2) clustering consecutive above-threshold windows
      3) sampling inside each cluster every K windows (NOT 1 per cluster)
      4) optional refractory on event times (min_distance_sec)

    Returns: list of (t_start, t_end) intervals in seconds (timestamps units).
    """

    # --- sanitize inputs ---
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    ws = np.asarray(window_starts, dtype=np.int64)

    n = min(len(p), len(ws))
    p = p[:n]
    ws = ws[:n]

    # --- threshold mask ---
    keep_mask = (p >= float(threshold))
    idxs = np.flatnonzero(keep_mask)
    # An prob < threshold την πετάει

    if idxs.size == 0:
        if debug:
            print(f"[DEBUG] No windows >= threshold={threshold}")
        return []

    #build clusters of consecutive indices
    clusters = []
    start = idxs[0]
    prev = idxs[0]
    for idx in idxs[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            clusters.append((start, prev))
            start = idx
            prev = idx
    clusters.append((start, prev))
    # Ενώνει τα συνεχόμενα παράθυρα σε clusters

    K = int(K)
    if K < 1:
        K = 1

    keep = []
    for a, b in clusters:
        keep.extend(range(a, b + 1, K))   #  Σε κάθε cluster, παίρνει ένα κάθε K winds.

    # unique + sorted
    keep = sorted(set(keep))

    # min_distance_sec on event times ---
    if float(min_distance_sec) > 0.0:
        # event time = window END time
        end_idx = ws[keep] + (window_size - 1)
        end_idx = np.clip(end_idx, 0, len(timestamps) - 1)
        event_times = np.asarray(timestamps)[end_idx]

        filtered = []
        last_t = -np.inf
        for j, t in zip(keep, event_times):
            if (t - last_t) >= float(min_distance_sec): # 2 ανιχνεύσεις κοντα αγνοεί την 2η
                filtered.append(j)
                last_t = t
        keep = filtered

    # --- build detected intervals ---
    detected_intervals = []
    ts = np.asarray(timestamps)

    for j in keep:
        start_idx = int(ws[j])
        end_idx = start_idx + window_size - 1
        if start_idx < 0 or end_idx >= len(ts):
            continue
        detected_intervals.append((float(ts[start_idx]), float(ts[end_idx])))

    if debug:
        print(f"[DEBUG] threshold={threshold} | clusters={len(clusters)} | kept={len(keep)} | K={K} | min_distance_sec={min_distance_sec}")

    return detected_intervals



    #     {
    #     "f1_score": f1,
    #     "TP": TP,
    #     "FP": FP,
    #     "FN": FN,
    #     "precision": precision,
    #     "recall": recall
    #
    # }
