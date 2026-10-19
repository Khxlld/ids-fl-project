from __future__ import annotations

import numpy as np
import pandas as pd

from run_rounds23_diagnostic import _changes, _class_metric_changes, _summary


def test_summary_reports_distribution_quantiles() -> None:
    observed = _summary(np.asarray([0.0, 1.0, 2.0, 3.0]))

    assert observed["rows"] == 4
    assert observed["mean"] == 1.5
    assert np.isclose(observed["sample_sd"], np.std([0.0, 1.0, 2.0, 3.0], ddof=1))
    assert observed["p50"] == 1.5
    assert observed["min"] == 0.0
    assert observed["max"] == 3.0


def test_distribution_changes_use_round_zero_and_previous_round() -> None:
    frame = pd.DataFrame(
        [
            {
                "seed": 1,
                "mode": "iid",
                "true_class": "Normal Traffic",
                "measure": "margin",
                "round": round_index,
                **{name: value for name in ("mean", "sample_sd", "min", "p05", "p25", "p50", "p75", "p95", "max")},
            }
            for round_index, value in ((0, 1.0), (1, 1.5), (2, 1.25))
        ]
    )

    observed = _changes(frame, ["seed", "mode", "true_class", "measure"])

    assert np.isnan(observed.loc[0, "mean_change_from_previous_round"])
    assert observed.loc[1, "mean_change_from_round0"] == 0.5
    assert observed.loc[2, "mean_change_from_round0"] == 0.25
    assert observed.loc[2, "mean_change_from_previous_round"] == -0.25


def test_class_metric_changes_keep_thresholded_and_closed_set_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "seed": 1,
                "mode": "iid",
                "variant": variant,
                "class_name": "Sybil Attack",
                "round": round_index,
                "recall": value,
                "precision": value,
                "prediction_count": value * 100,
            }
            for variant, values in (
                ("closed_set", (0.2, 0.4)),
                ("thresholded", (0.1, 0.15)),
            )
            for round_index, value in enumerate(values)
        ]
    )

    observed = _class_metric_changes(frame)
    closed_round1 = observed[(observed["variant"] == "closed_set") & (observed["round"] == 1)].iloc[0]
    thresholded_round1 = observed[(observed["variant"] == "thresholded") & (observed["round"] == 1)].iloc[0]

    assert np.isclose(closed_round1["recall_change_from_round0"], 0.2)
    assert np.isclose(thresholded_round1["recall_change_from_round0"], 0.05)
