# Statistics

This document describes `paretoguard.statistics` (Phase F, Commit 30):
confidence intervals, hypothesis tests, effect sizes, multiple-comparison
correction, Pareto frontier analysis, and the typed `ExperimentComparison`
result that ties them together. Regression detection (Commit 31) and report
generation (Commit 32) are documented separately and build on this module.

Every statistic here answers a real experiment question this repo actually
asks — see each function's own docstring for *why* it exists, not just what
it computes (per CLAUDE.md: don't add capability for its own sake).

## Bootstrap confidence intervals

`statistics.bootstrap.bootstrap_ci`/`bootstrap_proportion_ci` implement the
**percentile method**: resample the observed data with replacement
`n_resamples` times (default 2,000), compute the statistic on each
resample, and report the `alpha/2`/`1 - alpha/2` percentiles of that
resampled distribution as the interval.

- **Determinism**: every resample draw comes from a `numpy.random.Generator`
  seeded explicitly (never global numpy random state) — the same data and
  seed always produce a bit-for-bit identical interval.
- **Limitation, not hidden**: below `MIN_RELIABLE_N` (30) observations, the
  percentile method is known to be biased/unstable. `BootstrapResult
  .reliable` is `False` in that case — the interval is still computed and
  returned (never silently withheld), but must not be presented as precise.
- This is the plain percentile method, not bias-corrected-and-accelerated
  (BCa). BCa is more accurate for skewed statistics at small-to-moderate n
  but meaningfully more complex to implement correctly; the known small-n
  bias of the percentile method is exactly why `reliable` exists rather
  than pretending the interval is always trustworthy.
- `n_resamples=2000` is a general-purpose default, not tuned per metric.

## Choosing a significance test: paired vs. independent

This is the single most important methodological decision in this module,
and it is made by inspecting the data, not by a fixed default (see the
Phase F spec: "before implementation, inspect the experiment design and
choose the statistically correct formulation").

**Two configurations evaluated over the identical task IDs, in the
identical order, are paired** — the normal case in this repo (e.g. two
`resilience_benchmark` recovery configs run over the same `EvalSuite`).
Discarding that pairing and treating the two samples as independent throws
away information and typically under-powers the comparison for a
task-level effect.

| Metric type | Paired (same task IDs) | Independent (different task IDs) |
|---|---|---|
| Binary (success/fail) | **McNemar exact test** (binomial test on discordant pairs) | **Fisher's exact test** (2x2 contingency table) |
| Continuous (cost, latency) | **Wilcoxon signed-rank** | **Mann-Whitney U** |

`statistics.comparison._paired` decides which regime applies: task ID lists
must be equal length and identical *in order*. Same IDs in a different
order are **not** treated as paired — position-by-position correspondence
cannot be assumed without it, and a silent mismatch would corrupt McNemar's
discordant-pair count or Wilcoxon's per-pair differences.

McNemar's test only uses the *discordant* pairs (where the two configs
disagree on one task) — concordant pairs (both succeeded or both failed)
carry no information about whether the outcome changed and are correctly
excluded, not double-counted.

Cost and latency use rank-based (non-parametric) tests rather than a
paired/independent t-test: this repo's cost and latency distributions are
typically right-skewed (a handful of slow/expensive outliers), so a t-test's
normality assumption is not assumed without checking.

## Effect sizes — always reported alongside a p-value, never instead of one

`statistics.effect_size` provides:

- **`risk_difference`**: absolute (percentage points) and relative change
  in a proportion. Relative difference is `None` (not fabricated as
  infinite) when the baseline rate is 0.
- **`odds_ratio`**: `None` whenever any cell of the underlying 2x2 table is
  zero — the raw odds ratio is then 0, undefined, or infinite, and this
  module never substitutes a continuity-corrected estimate without being
  asked to.
- **`continuous_delta`**: absolute and relative change in a mean (cost,
  latency).

This repo never reduces an analysis to "significant / not significant" —
every `ExperimentComparison` carries both a p-value *and* an effect size,
so a tiny but statistically significant change at huge `n` is visibly
distinguishable from a large, meaningful one (see Commit 31's practical-vs-
statistical-significance handling, which uses exactly this distinction).

## Multiple-comparison correction

`statistics.significance.holm_bonferroni` implements the Holm-Bonferroni
step-down procedure. It returns adjusted p-values in the *same order* as
the input — the family of hypotheses being corrected is exactly the
sequence passed in. Applying it across an unrelated mix of analyses (rather
than, say, every pairwise comparison within one flagship benchmark) would
over-correct hypotheses that were never actually part of one family;
callers are responsible for passing the correct family. `ExperimentComparison
.adjusted_p_value` is `None` unless a caller explicitly supplies one
(`compare_binary_outcomes`/`compare_continuous_outcomes`'s `adjusted_p_value`
parameter) — this module never silently corrects for a family it wasn't
told about.

## The `ExperimentComparison` result

`statistics.comparison.compare_binary_outcomes`/`compare_continuous_outcomes`
build a single typed `ExperimentComparison`: both estimates, both kinds of
delta, a bootstrap confidence interval on the delta itself (resampled
paired or independent to match the test choice), the test actually used,
raw and (optionally) adjusted p-values, an effect size, the sample size,
and a descriptive `ComparisonConclusion`:

- `IMPROVEMENT_DETECTED` / `REGRESSION_DETECTED` — statistically significant
  (after adjustment, if supplied) *and* the delta's sign matches/opposes
  `higher_is_better`.
- `NO_CLEAR_DIFFERENCE` — not statistically significant, or the delta is
  exactly zero.
- `INSUFFICIENT_EVIDENCE` — either arm has fewer than
  `MIN_SAMPLE_SIZE_FOR_EVIDENCE` (10) observations, **regardless of the
  p-value**. A "significant" result from a handful of observations is not
  evidence this repo reports as such.

Conclusions are deliberately descriptive, never causal: `IMPROVEMENT_DETECTED`
means "candidate's measured estimate was higher (or lower, for a
lower-is-better metric) than baseline's, by a statistically distinguishable
margin" — it does not mean, and must never be read as, "config X caused the
improvement."

## Pareto frontier

`statistics.pareto_frontier.analyze_pareto_frontier` reuses
`routing.pareto`'s dominance algorithm (the same one `ParetoRouter` uses at
routing time) rather than reimplementing it — a candidate that dominates
another for routing purposes is dominated the same way when reporting on
measured results after the fact. Never declares one configuration
universally "best": the result is a frontier (every non-dominated tradeoff)
and, for every dominated point, exactly which frontier member(s) dominate
it — an explanation, not just a verdict. Latency is optional; when omitted,
every point is given latency `0.0` (excluded from dominance) so a caller
reporting only cost/success doesn't need to fabricate latency data.
