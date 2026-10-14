# Thesis to code

Figure numbers are those of the thesis. "Result key" is the path inside the stored JSON. The figures of
Chapters 5 to 7 that come from the studies are listed in the [reproducing page](reproducing.md#6-figure-by-figure-script-and-stored-result) and in
`figures/MANIFEST.md`.

| Thesis item | Drawn or computed by | Simulation script | Stored result (key) |
|---|---|---|---|
| Fig. 3.1 isolated outcrop | `lunarmag.figures.geometry_figs.draw_outcrop_geometry` | constants as in `analysis/a02`, `a09` | none (geometry) |
| Fig. 3.2 skate park | `...geometry_figs.draw_skatepark_geometry` | `engine/skatepark_1cm_dr_sweep.outcrop_configs_for` | none (scene definition) |
| Fig. 3.3 three-layer wall | `...geometry_figs.draw_threelayer_geometry` | `inversion_helpers.three_layer_mesh` | none (geometry) |
| Fig. 4.1 recovery vs field strength | `...results_figs.draw_detection_envelope` | `analysis/a02b_detection_sweep_N1000.py` | `a02b_detection_sweep_N1000.json` `detection_sweep/drive_around[*]/{dir_err_deg,dM_abs_pct}/median` |
| Fig. 4.2 imaging mis-pose | `...results_figs.draw_lidar_sensitivity` | `analysis/a02_isolated_outcrop_paper_geometry.py` | `a02_...json` `lidar_grid/<survey>[*]/typical/*/median` |
| Fig. 4.3 skate park recovery | `...results_figs.draw_skatepark_recovery` | `engine/skatepark_1cm_dr_sweep.py` | `skatepark_1cm_dr_sweep_results.json` `sweep/{A,B}/*` |
| Fig. 4.4 second traverse height | `...results_figs.draw_second_traverse_height` | `engine/scripts/elevated_traverse_height_sweep.py` | `elevated_traverse_height_sweep_results.json` `heights/*/per_layer_closed_form_ls` |
| Fig. 5.1 correlated noise and drift | `...results_figs.draw_noise_requirement` | `analysis/a06_isolated_robustness.py` parts A, G | `a06_...json` `A_dc_offsets/drive_around/A3_correlated`, `G_gradient_and_drift/drive_around/drift_L100m` |
| Fig. 5.2 combined error budget | `...results_figs.draw_budget` | `analysis/a09_systematics.py` parts Z, U, V, X, Y, G, q, f, g, h | `a09_systematics.json` `<case>/<direction>/{all_sources,noise_free_bias}/dir_err_deg/mean` |
| Table 3.1 noise budget | arithmetic in the text | none | none |
| Table 4.1 inversion approaches | `engine/inversion_approach_comparison.py` | same | `inversion_approach_comparison_results.json` `summary` |
| Table 4.2 robustness | `analysis/a06_isolated_robustness.py` parts A, B | same | `a06_...json` `A_dc_offsets`, `B_direction_height_scan` |
| Table 4.3 three-layer recovery | `engine/three_layer_wall.py` | same | `three_layer_wall_results.json` `arms/*/per_layer_*` |
| Table 5.1 error budget | `analysis/a09_systematics.py` | same | `a09_systematics.json` (same keys as Fig. 5.2) |
| Peak field 0.25 nT (Section 3.2.1) | `physics.prism_field` (closed form) | `analysis/a02` | `a02_...json` `baseline/drive_around/peak_field_nT` = 0.2459 |
| Drive-around 3.0 deg and 3.7% (medians), mean 3.2 deg, p95 6.1 deg | `estimators.predicted_mean_direction_error_deg` (prediction) | `analysis/a02` (1000 inversions) | `baseline/drive_around/dir_err_deg/{median,mean,p95}`, `dM_abs_pct/median` |
| Drive-by 6.6 deg and 7.3% | | `analysis/a02` | `baseline/drive_by/...` |
| Mis-pose: at most 5.4 deg and 10.6% (drive-around) | | `analysis/a02` | `lidar_grid/drive_around` (max of medians) |
| Approach comparison 0.27, 1.24, 3.10 deg | | `engine/inversion_approach_comparison.py` | `summary/<method>/dir_err_deg_mean` |
| Ground traverse: flows 3.7, 28.1, 33.4 deg | | `engine/three_layer_wall.py` | `arms/A_single_base_traverse/per_layer_closed_form_ls` |
| Elevated traverse: all flows within 5 deg for z2 = 6.4-13.2 m | | `engine/scripts/elevated_traverse_height_sweep.py` | `headline/thresholds/closed_form_ls/...` (upper crossing interpolated) |
| Requirement set with both mitigations: 3.0-4.9 deg | | `analysis/a09` parts f, g, h | `REQF_*` |

`lunarmag numbers [section]` prints all 374 numbers that the thesis shares with the companion paper, each next to
its sentence. Table A.1 of the thesis (Appendix A) lists the scripts and result files behind each result.
