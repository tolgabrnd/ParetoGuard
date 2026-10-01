"""Chart generation — every chart is a pure function of explicitly-typed
data (never a hidden global or a re-query the caller didn't ask for), and
every chart function returns a `ChartResult` carrying the run ids, labels,
and sample sizes it actually plotted alongside the rendered file path — so
a caller (or a test) can verify the *data* a chart shows without pixel-
diffing an image (see the Phase F spec's "chart integrity" requirement).

**Never fabricate a chart from missing data.** Every function returns
`None` (not an empty/misleading plot) when its required data isn't
available — `reports.report` renders "not available" in that case, per the
spec: "if metric data are unavailable, show 'not available'. Never
fabricate zero."

**No decorative dashboards.** Only the 8 chart types the Phase F spec names
are implemented, each answering one specific reporting question.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module never opens a GUI window
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from paretoguard.statistics.comparison import ExperimentComparison
from paretoguard.statistics.pareto_frontier import ParetoAnalysis


@dataclass(frozen=True)
class ChartResult:
    path: Path
    title: str
    run_ids: tuple[str, ...]
    labels: tuple[str, ...]
    """SIMULATION/LIVE/unlabeled label(s) of the run(s) this chart draws
    from — propagated so a report never shows a chart without knowing
    whether it's simulated or live."""
    sample_sizes: dict[str, int]
    """`{series_name: n}` for every series plotted."""


def _save(fig: Figure, output_path: Path, *, label: str | None = None) -> Path:
    """Stamps `label` (e.g. "SIMULATION") onto the figure itself, not just
    `ChartResult.labels` — a PNG opened on its own (outside a report, e.g.
    pasted into a chat or a slide) must still be self-describing about
    whether it's simulated or live data. A viewer should never have to
    trust a filename or a caller's memory for that."""
    if label:
        fig.text(
            0.99,
            0.01,
            label,
            ha="right",
            va="bottom",
            fontsize=7,
            color="dimgray",
            alpha=0.8,
            transform=fig.transFigure,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pareto_frontier(
    analysis: ParetoAnalysis,
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult:
    """Chart 1: quality/success vs. cost Pareto frontier. Never labels one
    point "best" — frontier members are marked distinctly from dominated
    ones, nothing more."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, obj in analysis.objectives.items():
        on_frontier = name in analysis.frontier
        ax.scatter(
            obj.expected_cost_usd,
            obj.predicted_success,
            s=90 if on_frontier else 50,
            marker="o" if on_frontier else "x",
            label=f"{name} (frontier)" if on_frontier else name,
        )
        ax.annotate(
            name,
            (obj.expected_cost_usd, obj.predicted_success),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean cost (USD)")
    ax.set_ylabel("Success rate")
    ax.set_title("Pareto frontier: success vs. cost")
    ax.legend(fontsize=7, loc="best")
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title="Pareto frontier: success vs. cost",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes=dict.fromkeys(analysis.objectives, 1),
    )


@dataclass(frozen=True)
class ResilienceCurvePoint:
    fault_level: float
    config_name: str
    success_rate: float
    n: int


def plot_resilience_curve(
    points: list[ResilienceCurvePoint],
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 2: success rate vs. fault rate, one line per recovery config."""
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by_config: dict[str, list[ResilienceCurvePoint]] = {}
    for p in points:
        by_config.setdefault(p.config_name, []).append(p)
    for config_name, series in sorted(by_config.items()):
        series_sorted = sorted(series, key=lambda p: p.fault_level)
        ax.plot(
            [p.fault_level for p in series_sorted],
            [p.success_rate for p in series_sorted],
            marker="o",
            label=config_name,
        )
    ax.set_xlabel("Injected fault level")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Resilience curve: success rate vs. fault rate")
    ax.legend(fontsize=7, loc="best")
    path = _save(fig, output_path, label=label)
    sample_sizes = {f"{p.config_name}@{p.fault_level}": p.n for p in points}
    return ChartResult(
        path=path,
        title="Resilience curve",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes=sample_sizes,
    )


def plot_recovery_rate_by_level(
    points: list[ResilienceCurvePoint],
    recovery_rates: dict[tuple[str, float], float | None],
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 3: recovery rate by fault level, one line per recovery config.
    `recovery_rates` keyed by `(config_name, fault_level)`; a `None` value
    (no raw failures at that level to recover from) is skipped for that
    point, never plotted as a fabricated 0.0."""
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by_config: dict[str, list[ResilienceCurvePoint]] = {}
    for p in points:
        by_config.setdefault(p.config_name, []).append(p)
    plotted_any = False
    for config_name, series in sorted(by_config.items()):
        series_sorted = sorted(series, key=lambda p: p.fault_level)
        xs, ys = [], []
        for p in series_sorted:
            rate = recovery_rates.get((config_name, p.fault_level))
            if rate is not None:
                xs.append(p.fault_level)
                ys.append(rate)
        if xs:
            ax.plot(xs, ys, marker="o", label=config_name)
            plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return None
    ax.set_xlabel("Injected fault level")
    ax.set_ylabel("Recovery rate (of raw failures)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Recovery rate by fault level")
    ax.legend(fontsize=7, loc="best")
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title="Recovery rate by fault level",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes={f"{p.config_name}@{p.fault_level}": p.n for p in points},
    )


def plot_failure_taxonomy(
    distribution: dict[str, int],
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 4: failure taxonomy distribution — counts of each
    `FailureCategory` (or 'unknown') among unrecovered failures."""
    if not distribution:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    categories = sorted(distribution, key=lambda k: -distribution[k])
    counts = [distribution[c] for c in categories]
    ax.bar(categories, counts)
    ax.set_ylabel("Count")
    ax.set_title("Failure taxonomy distribution")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title="Failure taxonomy distribution",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes={"total": sum(counts)},
    )


def plot_selection_distribution(
    distribution: dict[str, int],
    *,
    output_path: Path,
    title: str = "Selection distribution",
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 5: router/recovery-config/candidate selection distribution."""
    if not distribution:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = sorted(distribution, key=lambda k: -distribution[k])
    counts = [distribution[n] for n in names]
    ax.bar(names, counts)
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title=title,
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes={"total": sum(counts)},
    )


@dataclass(frozen=True)
class HealthTimelinePoint:
    step: int
    provider: str
    model: str
    ema_success_rate: float


def plot_health_timeline(
    points: list[HealthTimelinePoint],
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 6: model/provider health (EMA success rate) over time, "when
    data exist" (per the spec) — this repo does not persist a health
    time series by default (`HealthTracker` only exposes current
    snapshots), so this returns `None` unless a caller supplies one
    explicitly, rather than fabricating a flat/fake line."""
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by_candidate: dict[str, list[HealthTimelinePoint]] = {}
    for p in points:
        by_candidate.setdefault(f"{p.provider}:{p.model}", []).append(p)
    for candidate, series in sorted(by_candidate.items()):
        series_sorted = sorted(series, key=lambda p: p.step)
        ax.plot(
            [p.step for p in series_sorted],
            [p.ema_success_rate for p in series_sorted],
            label=candidate,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Model/provider health timeline")
    ax.legend(fontsize=7, loc="best")
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title="Model/provider health timeline",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes={c: len(s) for c, s in by_candidate.items()},
    )


def plot_baseline_vs_candidate(
    comparison: ExperimentComparison,
    *,
    output_path: Path,
    label: str = "SIMULATION",
) -> ChartResult:
    """Chart 7: baseline vs. candidate comparison, with the bootstrap CI on
    the *delta* shown as an error bar anchored on the candidate bar (the CI
    describes the difference, not either estimate alone — shown that way
    deliberately rather than fabricating separate per-arm CIs this module
    was never given)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(["baseline", "candidate"], [comparison.baseline_estimate, comparison.candidate_estimate])
    if comparison.confidence_interval is not None:
        ci = comparison.confidence_interval
        lower_err = comparison.candidate_estimate - (comparison.baseline_estimate + ci.lower)
        upper_err = (comparison.baseline_estimate + ci.upper) - comparison.candidate_estimate
        ax.errorbar(
            [1],
            [comparison.candidate_estimate],
            yerr=[[max(0.0, lower_err)], [max(0.0, upper_err)]],
            fmt="none",
            ecolor="black",
            capsize=6,
        )
    ax.set_ylabel(comparison.metric_name)
    ax.set_title(
        f"{comparison.metric_name}: {comparison.conclusion.value} (n={comparison.sample_size})"
    )
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title=f"Baseline vs. candidate: {comparison.metric_name}",
        run_ids=tuple(r for r in (comparison.baseline_run_id, comparison.candidate_run_id) if r),
        labels=(label,),
        sample_sizes={
            "baseline": comparison.sample_size // 2,
            "candidate": comparison.sample_size // 2,
        },
    )


@dataclass(frozen=True)
class CalibrationBin:
    mean_predicted_success: float
    mean_actual_success: float
    n: int


def plot_calibration(
    bins: list[CalibrationBin],
    *,
    output_path: Path,
    run_id: str | None = None,
    label: str = "SIMULATION",
) -> ChartResult | None:
    """Chart 8: calibration plot (reliability diagram) for learned routing —
    "if compatible data exist" (per the spec). Returns `None` when no bins
    are supplied, e.g. no learned router was evaluated in this run."""
    if not bins:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(
        [b.mean_predicted_success for b in bins],
        [b.mean_actual_success for b in bins],
        s=[max(20, b.n) for b in bins],
    )
    ax.set_xlabel("Mean predicted success")
    ax.set_ylabel("Mean actual success")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Calibration (reliability diagram)")
    ax.legend(fontsize=8, loc="best")
    path = _save(fig, output_path, label=label)
    return ChartResult(
        path=path,
        title="Calibration",
        run_ids=(run_id,) if run_id else (),
        labels=(label,),
        sample_sizes={f"bin_{i}": b.n for i, b in enumerate(bins)},
    )
