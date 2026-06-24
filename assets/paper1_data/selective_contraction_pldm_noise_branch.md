# Selective-Contraction Branch Table

Scope: existing PLDM sweep. This is a branch diagnostic, not a new main claim.

| Task | best std | obs-noise 0.08 success | encoder radius R_E | prediction radius R_F | original NN L2 | transition L2 | aux ADM | aux SPRR | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TwoRoom | 0.06 | 67 -> 98 | 3.29 -> 0.235 | 0.611 -> 0.0339 | 4.52 -> 4.62 | 16.2 -> 16.5 | 20.8 -> 20.8 | 1.18 -> 26.3 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| PushT | 0.03 | 18.3 -> 73 | 0.874 -> 0.153 | 0.846 -> 0.0907 | 7.99 -> 8.13 | 10.2 -> 10.3 | 21.6 -> 21.5 | 1.26 -> 12.9 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| Reacher | 0.03 | 80.3 -> 81.7 | 0.237 -> 0.135 | 0.0597 -> 0.0386 | 5.46 -> 5.08 | 11.2 -> 10.6 | 21.8 -> 21.6 | 18.6 -> 27.5 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| Cube | 0.04 | 48.3 -> 54.7 | 0.831 -> 0.146 | 0.453 -> 0.0539 | 7.48 -> 7.16 | 14 -> 13.9 | 21.5 -> 21.6 | 1.35 -> 9.17 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |

Reading: lower R_E/R_F means smaller same-state perturbation spread in the reported feature space. Higher SPRR means the auxiliary action-distance margin is larger relative to paired rollout disagreement. ADM/SPRR come from the exploratory observation+goal Phase-0 diagnostic, so they are supportive visualization/branch evidence only.

Visualization note: selective-contraction cluster plots should be read through the high-D panel statistics. The 2-D t-SNE envelopes are qualitative summaries of repeated same-state perturbation samples, not estimates of the true high-D basin boundary.
