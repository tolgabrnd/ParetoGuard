**SIMULATION — NOT LIVE PROVIDER PERFORMANCE**

# Benchmark report: resilience-resilience_v1-0.3-E-full_policy

## Run identity (reproducibility)
- run_id: `resilience-resilience_v1-0.3-E-full_policy`
- label: SIMULATION
- git_sha: not available
- paretoguard_version: 0.1.0
- suite: resilience_v1 v1.0.0
- seed: 0
- router: static
- pricing_config_version: not available
- os / python: Windows / 3.12.14
- sample size: 30

## Metrics
- success rate: 1.000 (95% CI [1.000, 1.000])
- cost per successful task: not available
- mean latency (successful tasks): 0.02 ms

## Failure taxonomy
- no failures recorded

## Recovery / circuit-breaker outcomes
- raw failure rate: 0.333
- recovery rate: 1.000
- retry rate: 0.300
- fallback rate: 0.167
- escalation rate: 0.000
- probe rate: 0.000
- average attempts: 1.47

## Pareto analysis
- not available

## Comparison vs. baseline
- baseline_run_id: `resilience-resilience_v1-0.3-A-no_recovery`
- metric: success_rate
- baseline: 0.6667, candidate: 1.0000
- absolute delta: +0.3333
- test: mcnemar_exact (paired=True, p=0.001953)
- conclusion: improvement_detected
(descriptive only — not a causal claim)

## Limitations
- No cost data available for this run (no PricingTable configured, or no successes).
- No Pareto analysis available: a single run's EvalResults don't carry the multi-candidate data a frontier needs (see evals.matrix.MatrixRunner).