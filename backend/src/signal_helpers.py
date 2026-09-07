"""
Pure-logic signal processing helpers for video_processing.py.

Extracted to a separate module so they can be unit-tested without
the heavy cv2/modal/scenedetect dependency chain.
"""

import logging
import numpy as np
import scipy.ndimage as ndimage

_logger = logging.getLogger(__name__)

# Face-detection bounding boxes jitter by ~0.02-0.06 in normalised coords even for
# perfectly stationary speakers.  A speaker genuinely walking across frame produces
# std > 0.15.  Lowering this to 0.06 allows tracking to activate more willingly.
STATIONARY_STD_THRESHOLD = 0.06

# When the ASD model alternates between two stationary faces in a wide shot,
# the positions cluster into groups. A reduced gap threshold avoids improperly
# merging adjacent speakers.
CLUSTER_GAP = 0.04


def smooth_segment(raw: np.ndarray, default: float, sigma: int) -> np.ndarray:
    """
    Stabilises a 1-D position array (gaps = -1) by choosing between a 
    STABLE LOCKED shot (zero movement) or a SMOOTH TRACKING shot.
    """
    seg_len = len(raw)
    valid_mask = raw != -1

    if not np.any(valid_mask):
        return np.full(seg_len, default)

    valid_points = raw[valid_mask]
    
    # ── 1. Identify Clusters (Stationary Subjects) ──────────────────────────
    sorted_pts = np.sort(valid_points)
    gaps = np.diff(sorted_pts)
    gap_indices = np.where(gaps > CLUSTER_GAP)[0]
    
    boundaries = np.concatenate([[-1], gap_indices, [len(sorted_pts) - 1]])
    clusters = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i]) + 1
        end = int(boundaries[i + 1]) + 1
        clusters.append(sorted_pts[start:end])

    # ── 2. Determine if the scene is stationary ────────────────────────────
    # A scene is stationary if the largest cluster is stable.
    largest_cluster = max(clusters, key=len)
    is_stationary = float(np.std(largest_cluster)) < STATIONARY_STD_THRESHOLD

    if is_stationary:
        # LOCKED MODE: No micro-movements.
        # We assign each frame to its nearest cluster center to handle 
        # speaker-switches (snap-cuts) while remaining perfectly stationary.
        cluster_centers = [float(np.median(c)) for c in clusters]
        out = np.full(seg_len, default)
        
        # Initial position from first valid target (not just the left-most person)
        valid_idx = np.where(raw != -1)[0]
        if len(valid_idx) > 0:
            first_target = float(raw[valid_idx[0]])
            last_pos = min(cluster_centers, key=lambda c: abs(c - first_target))
        else:
            last_pos = cluster_centers[0]
        
        for i in range(seg_len):
            if raw[i] != -1:
                target = float(raw[i])
                best_center = min(cluster_centers, key=lambda c: abs(c - target))
                
                # Check if we are doing a snap-cut (speaker switch)
                if abs(last_pos - best_center) > CLUSTER_GAP:
                    # Snap immediately to the new speaker's approximate center
                    last_pos = best_center
                    
            out[i] = last_pos
            
        # Enforce minimum shot duration (1.2 seconds = 30 frames @ 25fps)
        # Prevents seizure-inducing flickering when TalkNet rapidly alternates between speakers
        MIN_SHOT = 30
        stabilized = out.copy()
        run_start = 0
        for i in range(1, seg_len):
            if out[i] != out[i-1]:
                if (i - run_start) < MIN_SHOT:
                    if run_start > 0:
                        # This shot was too short, revert it to the previous speaker
                        stabilized[run_start:i] = stabilized[run_start-1]
                    else:
                        # The very first shot was too short, overwrite it with the incoming speaker
                        stabilized[run_start:i] = out[i]
                run_start = i
                
        # Handle the final run
        if (seg_len - run_start) < MIN_SHOT and run_start > 0:
            stabilized[run_start:] = stabilized[run_start-1]
            
        return stabilized

    # ── 3. TRACKING MODE: Moving subject ──────────────────────────────────
    # Gap-fill then Gaussian smooth.
    filled = raw.copy()
    idxs = np.arange(seg_len)
    first_valid = int(np.argmax(valid_mask))
    filled[:first_valid] = filled[first_valid]
    last_valid = int(seg_len - 1 - np.argmax(valid_mask[::-1]))
    filled[last_valid + 1:] = filled[last_valid]

    valid_mask_filled = filled != -1
    if not np.all(valid_mask_filled):
        filled[~valid_mask_filled] = np.interp(
            idxs[~valid_mask_filled], idxs[valid_mask_filled], filled[valid_mask_filled]
        )

    effective_sigma = min(sigma, max(1, seg_len // 4))
    return ndimage.gaussian_filter1d(filled, sigma=effective_sigma)


def stabilize_segment(raw: np.ndarray, min_entry: int, min_gap: int) -> np.ndarray:
    """
    Two-pass stabiliser for a boolean signal within one scene segment.
      Pass 1: Remove True-runs shorter than min_entry frames (noise).
      Pass 2: Merge adjacent True-runs separated by < min_gap False frames (hysteresis).
    """
    n = len(raw)
    if n == 0:
        return raw.copy()
    runs = []
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_entry]
    if not runs:
        return np.zeros(n, dtype=bool)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if (s - merged[-1][1] - 1) < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    result = np.zeros(n, dtype=bool)
    for s, e in merged:
        result[s: e + 1] = True
    return result


def stabilize_speaker_identity(
    raw_cx: np.ndarray,
    min_frames: int,
    px_threshold: float,
) -> np.ndarray:
    """
    Debounce a per-frame face-centre (cx) signal for a single speaker slot.

    Problem: when the active speaker switches to a different camera angle
    (a cut to a new face or a wide-to-close zoom), TalkNet immediately assigns
    speaking=True to the new face.  The raw cx signal therefore jumps by
    hundreds of pixels in a single frame.  Without this filter, the Gaussian
    smoother receives that jump and starts drifting toward the new position
    immediately, causing the crop to visibly slide during what might be a
    transient one-second insert that never returns.

    Solution: treat the raw cx signal as a "candidate".  The crop only commits
    to a new position when the candidate has been consistently different from
    the committed position for at least min_frames consecutive frames.  Until
    then, the output is filled with the last committed position.

    Args:
        raw_cx:       1-D array of face-centre x values, -1 where no face.
        min_frames:   Minimum consecutive frames the new position must be held
                      before the switch is committed (e.g. 15 = ~0.6 s @ 25 fps).
        px_threshold: Minimum pixel displacement between old and new centre that
                      counts as a "different position" worth debouncing.

    Returns:
        Stabilised cx array (same shape as raw_cx, -1 entries preserved).
    """
    n = len(raw_cx)
    if n == 0:
        return raw_cx.copy()

    out = raw_cx.copy()

    # Find the first valid sample as the initial committed position.
    committed_cx: float = -1.0
    for i in range(n):
        if raw_cx[i] != -1:
            committed_cx = float(raw_cx[i])
            break

    if committed_cx == -1.0:
        return out  # no valid samples at all

    candidate_cx: float = committed_cx
    candidate_run: int = 0

    for i in range(n):
        cx = float(raw_cx[i])

        if cx == -1.0:
            # Gap frame — hold the committed position in output.
            out[i] = committed_cx
            # Gap resets the candidate run (we lost sight of the new face).
            candidate_run = 0
            candidate_cx = committed_cx
            continue

        if abs(cx - committed_cx) < px_threshold:
            # Still on the committed face (or very close) — accept immediately.
            committed_cx = cx
            out[i] = committed_cx
            candidate_run = 0
            candidate_cx = committed_cx
        else:
            # Different position — start or continue a candidate run.
            if abs(cx - candidate_cx) < px_threshold:
                # Same candidate as before — extend the run.
                candidate_run += 1
            else:
                # Yet another new position — reset candidate.
                candidate_cx = cx
                candidate_run = 1

            if candidate_run >= min_frames:
                # Candidate has been stable long enough — commit the switch.
                # Back-fill the frames we were debouncing so the cut happens
                # exactly when the angle changed, not 'min_frames' later.
                start_of_run = i - min_frames + 1
                out[start_of_run : i + 1] = candidate_cx
                
                _logger.debug(
                    f"  stabilize_speaker_identity: committed switch "
                    f"{committed_cx:.0f}px → {candidate_cx:.0f}px "
                    f"(back-filled {min_frames} frames)"
                )
                committed_cx = candidate_cx
                candidate_run = 0
            else:
                # Not yet stable — hold the previous committed position.
                out[i] = committed_cx

    return out
