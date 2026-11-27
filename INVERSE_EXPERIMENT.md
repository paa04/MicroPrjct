# Inverse spatial competition — experiments on actual repository data

Run from the root with Node 18+; no npm dependencies.

```bash
node fast_grid_experiment.js
node housing_experiment.js
node venue_experiment.js
node radius_experiment.js
node ensemble_experiment.js
node robust_local.js
node test_inverse.js
```

All measured reports are committed in `results/inverse_experiment/`. The original 100m `inverse_experiment.js` timed out in the interactive runner; the fast-grid and housing variants were actually run at 200m, and the ensemble at 100m candidate / 200m demand resolution. Full housing CSV was subsequently retrieved by blob SHA. The synthetic and real-report assertions were executed in the connector runtime; the equivalent standalone test is included.

## Historical prediction

28 training openings, 17 retrospective holdout openings, spatial grouping on training only. Each experiment selects configurations by training CV; the repeatedly examined 2025+ holdout is not a fresh independent test. Known true opening tile, contemporary POIs and incomplete historical competitor dates limit validity. Original baseline metrics are from the repository's temporal validation.

| Method | First within 300m | Any of 3 within 300m | Median nearest of 3 (m) |
|---|---:|---:|---:|
| Original joint, 100m | 4/17 | 10/17 | 230 |
| Original site-only, 100m | 3/17 | 10/17 | 223 |
| 200m inverse, store/office proxy | 2/17 | 9/17 | 252 |
| 200m inverse, full housing | 3/17 | 6/17 | 572 |
| Venue candidate model, site (CV-selected) | 1/17 | 8/17 | 342 |
| 200m radius likelihood, site (CV-selected) | 2/17 | 10/17 | 230 |
| 100m ensemble experiment, inverse_mass (CV-selected) | 4/17 | 7/17 | 370 |

None demonstrates better historical prediction than the original joint model. Do not select a winning variant by holdout metrics. Candidate coverage of venue model is 17/17 within 300m; ranking is the bottleneck, not missing locations. Housing improves coordinate RMSE vs site-only in the 200m comparison but does not improve top-three hit rate. Radius likelihood optimizes probability of a neighborhood rather than of an exact cell; report mean_nll is still exact-cell NLL, not the training objective. Optimization is bounded-iteration and not independently certified converged.

## Robust new opening in Maryino

1005 physically allowed/unoccupied grid candidates; 144 within 150m of a commercial/public footprint >=1000 m². 18 scenarios = 3 demand priors x 3 distance scales x 2 competitor sets (8 or 13 clubs). The economic criterion minimizes worst relative regret versus the best eligible venue-adjacent candidate in each scenario. The compromise restricts historical model mass within 300m to at least 50% of its peak among venue-adjacent candidates; this threshold is illustrative, not estimated.

- Minimax: 55.657144, 37.764679, OSM venue 51675639 (1500 m²), gain 8.82–24.76 pp, worst regret 4.57%.
- Maximin: 55.656246, 37.764679, same venue, min gain 8.83 pp.
- Historical site prediction: 55.650856, 37.742388, gain 5.28–9.22 pp, worst regret 77.86%.
- Compromise: 55.658042, 37.753534, Bratislavsky market, gain 8.32–18.14 pp, worst regret 45.42%.

These are normalized potential-demand shares with a scenario outside option, not sales or profit. Neither availability, actual usable floor space, lease, capacity, service quality, nor walking routes are verified in this experiment. Keep main unchanged; recommend checking the eastern candidate and Bratislavsky market in the existing pedestrian-graph model and obtaining verified premises data before making a business decision.
