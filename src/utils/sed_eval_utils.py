"""Segment-wise F1 evaluation for polyphonic SED using ``sed_eval``.

The paper reports the segment-wise F-score (Mesaros et al.) computed on
1-second segments. This module converts frame-level activity/probability
matrices into event lists and scores them with
``sed_eval.sound_event.SegmentBasedMetrics``.
"""

from typing import Dict, List

import numpy as np

# URBAN-SED classes (alphabetical, matching the dataset annotations).
URBAN_SED_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def median_filter(activity: np.ndarray, size: int) -> np.ndarray:
    """Temporal median filter over a (T, C) binary/real matrix."""
    if size <= 1:
        return activity
    from scipy.ndimage import median_filter as _mf  # local import; optional dep

    return _mf(activity, size=(size, 1))


def activity_to_events(
    activity: np.ndarray,
    hop_sec: float,
    classes: List[str],
    threshold: float = 0.5,
    median_filter_frames: int = 0,
) -> List[Dict]:
    """Convert a frame-level (T, C) matrix into a list of event dicts.

    Args:
        activity: (T, C) array of probabilities or binary activations.
        hop_sec: seconds per frame (feature hop length / sample rate).
        classes: class names indexed by column.
        threshold: binarisation threshold applied to probabilities.
        median_filter_frames: optional temporal median filter window (frames).

    Returns:
        List of ``{event_label, event_onset, event_offset}`` dicts.
    """
    binary = (activity >= threshold).astype(np.int8)
    if median_filter_frames and median_filter_frames > 1:
        try:
            binary = (median_filter(binary.astype(float), median_filter_frames) >= 0.5).astype(np.int8)
        except Exception:
            pass

    events: List[Dict] = []
    n_frames = binary.shape[0]
    for c, label in enumerate(classes):
        col = binary[:, c]
        onset = None
        for t in range(n_frames):
            if col[t] and onset is None:
                onset = t
            elif not col[t] and onset is not None:
                events.append(
                    {
                        "event_label": label,
                        "event_onset": onset * hop_sec,
                        "event_offset": t * hop_sec,
                    }
                )
                onset = None
        if onset is not None:
            events.append(
                {
                    "event_label": label,
                    "event_onset": onset * hop_sec,
                    "event_offset": n_frames * hop_sec,
                }
            )
    return events


def segment_based_f1(
    ref_events_per_file: List[List[Dict]],
    est_events_per_file: List[List[Dict]],
    classes: List[str],
    time_resolution: float = 1.0,
) -> Dict:
    """Compute segment-based metrics aggregated over files.

    Returns a dict with ``overall`` (f_measure/precision/recall as percentages)
    and ``class_wise`` (per-class f_measure as percentages).
    """
    import sed_eval
    import dcase_util

    metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )
    for ref, est in zip(ref_events_per_file, est_events_per_file):
        ref_c = dcase_util.containers.MetaDataContainer(ref)
        est_c = dcase_util.containers.MetaDataContainer(est)
        metrics.evaluate(reference_event_list=ref_c, estimated_event_list=est_c)

    results = metrics.results()

    def _pct(v):
        return 0.0 if v is None or np.isnan(v) else 100.0 * v

    overall = {
        "f_measure": _pct(results["overall"]["f_measure"]["f_measure"]),
        "precision": _pct(results["overall"]["f_measure"]["precision"]),
        "recall": _pct(results["overall"]["f_measure"]["recall"]),
        "error_rate": results["overall"]["error_rate"]["error_rate"],
    }
    class_wise = {}
    for label in classes:
        f = results["class_wise"][label]["f_measure"]["f_measure"]
        class_wise[label] = 100.0 * (0.0 if f is None or np.isnan(f) else f)

    avg_class_f1 = float(np.mean(list(class_wise.values()))) if class_wise else 0.0
    return {"overall": overall, "class_wise": class_wise, "average_class_f1": avg_class_f1}
