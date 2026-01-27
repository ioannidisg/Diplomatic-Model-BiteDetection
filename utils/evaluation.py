import numpy as np

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

    p = np.asarray(probs).reshape(-1).astype(np.float64)
    ws = np.asarray(window_starts, dtype=np.int64)

    n = min(len(p), len(ws))
    p = p[:n]
    ws = ws[:n]

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
