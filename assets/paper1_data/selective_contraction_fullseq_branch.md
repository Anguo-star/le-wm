# Selective-Contraction Branch Table

Scope: existing LeWM sweep. This is a branch diagnostic, not a new main claim.

| Task | best std | obs-noise 0.08 success | encoder radius R_E | prediction radius R_F | original NN L2 | transition L2 | aux ADM | aux SPRR | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TwoRoom | 0.08 | 65.7 -> 97.7 | 3.54 -> 0.235 | 0.785 -> 0.0284 | 3.66 -> 3.35 | 13.8 -> 12.6 | 19.8 -> 19.6 | 1.14 -> 36.8 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| PushT | 0.08 | 4.33 -> 89 | 1.73 -> 0.137 | 1.54 -> 0.0878 | 4.17 -> 4.07 | 5.89 -> 5.72 | 19.9 -> 19.8 | 1.23 -> 28.3 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| Reacher | 0.07 | 18.3 -> 84.7 | 4.24 -> 0.135 | 1.01 -> 0.0271 | 3.92 -> 3.72 | 7.53 -> 7.6 | 19.6 -> 19.6 | 1.35 -> 61.4 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |
| Cube | 0.03 | 47 -> 68.3 | 1.34 -> 0.238 | 0.957 -> 0.115 | 6.32 -> 6.38 | 9.62 -> 9.41 | 19.5 -> 19.6 | 1.04 -> 5.75 | same-state encoder/predictor radii contract; auxiliary ADM is preserved. |

Reading: lower R_E/R_F means smaller same-state perturbation spread in the reported feature space. Higher SPRR means the auxiliary action-distance margin is larger relative to paired rollout disagreement. ADM/SPRR come from the exploratory observation+goal Phase-0 diagnostic, so they are supportive visualization/branch evidence only.

Visualization note: selective-contraction cluster plots should be read through the high-D panel statistics. The 2-D t-SNE envelopes are qualitative summaries of repeated same-state perturbation samples, not estimates of the true high-D basin boundary.
