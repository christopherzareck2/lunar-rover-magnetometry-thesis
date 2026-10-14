# Figure manifest

Written by `make figures` (`lunarmag figures`). Every PDF in `thesis/MIT-thesis-template/figures/` is listed here, drawn in the thesis house style (`lunarmag.plotting`) and made PDF/A-1b ready (complete CIDSets, no transparency). `tests/test_figures.py` checks that the lunarmag figures plot exactly the values of the companion-paper scripts, that every figure PDF is PDF/A-1b clean, and that these hashes match the files in the thesis. Figure numbers are those of the thesis (kept from the previous manifest).

Environment: lunarmag 1.1.0.dev0, python 3.10.18, numpy 2.2.6, matplotlib 3.10.6.

| Fig. | File (thesis/MIT-thesis-template/figures/) | Drawn by | Stored inputs read (sha256) | PDF sha256 |
|---|---|---|---|---|
| 1.1 | `ch1_target_imagery.pdf` | `figures/scripts/fig_target_imagery.py` | `figures/inputs/nac_M113941548L.png` (9c50584674c3)<br>`figures/inputs/nac_M113941548L.json` (3f11e9157feb)<br>`figures/inputs/nac_M1345757107L.png` (f0c942db81c9)<br>`figures/inputs/nac_M1345757107L.json` (a60f8a055d1c)<br>`figures/inputs/apollo15_AS15-85-11451.png` (3a18ecffe394) | `664bf04acc164272` |
| 3.1 | `fig01_isolated_outcrop_geometry.pdf` | `src/lunarmag/figures/geometry_figs.py (draw_outcrop_geometry)` | geometry constants (as `analysis/a02`, `a09`) | `b7fec3b4ae52a0b5` |
| 3.2 | `fig02_skatepark_geometry.pdf` | `src/lunarmag/figures/geometry_figs.py (draw_skatepark_geometry)` | scene definition in `engine/skatepark_1cm_dr_sweep.py` | `6516f52757ae35e9` |
| 3.3 | `fig03_three_layer_geometry.pdf` | `src/lunarmag/figures/geometry_figs.py (draw_threelayer_geometry)` | `inversion_helpers.three_layer_mesh` and the traverse constants | `0c4d32d30243c7bc` |
| 4.1 | `fig04_detection_envelope.pdf` | `src/lunarmag/figures/results_figs.py (draw_detection_envelope)` | `results/analysis/a02b_detection_sweep_N1000.json` (a36a182de0af) | `af2afc79445b8087` |
| 4.2 | `fig05_lidar_sensitivity.pdf` | `src/lunarmag/figures/results_figs.py (draw_lidar_sensitivity)` | `results/analysis/a02_isolated_outcrop_paper_geometry.json` (5867c09a983c) | `b11aa2610019a147` |
| 4.3 | `fig06_skatepark_recovery.pdf` | `src/lunarmag/figures/results_figs.py (draw_skatepark_recovery)` | `results/engine/skatepark_1cm_dr_sweep_results.json` (ad5fb6d6b4d6) | `96f5331f590c3ced` |
| 4.4 | `fig07_second_traverse_height.pdf` | `src/lunarmag/figures/results_figs.py (draw_second_traverse_height)` | `results/engine/elevated_traverse_height_sweep_results.json` (e398042f9429) | `ca0e1668fe77c81f` |
| 5.1 | `fig08_noise_requirement.pdf` | `src/lunarmag/figures/results_figs.py (draw_noise_requirement)` | `results/analysis/a06_isolated_robustness.json` (afb6e8f7ed03) | `8439272dea3cddca` |
| 5.2 | `thesis_fig_budget.pdf` | `src/lunarmag/figures/results_figs.py (draw_budget)` | `results/analysis/a09_systematics.json` (4c1c137910d9) | `689322b1793be985` |
| 5.3 | `ch5_sobol_indices.pdf` | `studies/fig_sobol.py` | `studies/results/sobol.json` (904f1c12167c) | `441b2b5fd0c91785` |
| 5.4 | `ch5_flow_mean.pdf` | `studies/fig_flow_mean.py` | `studies/results/flow_mean.json` (1173c89690e2) | `f1680d41add6fa32` |
| 6.1 | `ch6_architecture.pdf` | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_architecture.py` (10cbba78c8a5) | `7a8db0d445bb1858` |
| 6.2 | `ch6_site_map.pdf` | `figures/scripts/fig_site_map.py` | `figures/inputs/lola_hillshade_nearside.png` (e7d0f1346a44)<br>`figures/inputs/site_coordinates.csv` (2d02cd012060) | `f082e9bd625721af` |
| 6.3 | `ch6_pareto.pdf` | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_data.py` (faab74564b54)<br>`results/analysis/a06_isolated_robustness.json` (afb6e8f7ed03)<br>`results/analysis/a09_systematics.json` (4c1c137910d9) | `24e5d03e2b22ad38` |
| 7.1 | `ch7_burial.pdf` | `ml/fig_ml.py` | `ml/results/sbi_results.json` (4ced5510c33e) | `a364674eb69b94f1` |
| not included | `ch6_conops.pdf` | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_data.py` (faab74564b54)<br>`results/analysis/a06_isolated_robustness.json` (afb6e8f7ed03)<br>`results/analysis/a09_systematics.json` (4c1c137910d9) | `6b7eb75359dac46b` |
| not included | `ch6_surface_time.pdf` | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_data.py` (faab74564b54)<br>`results/analysis/a06_isolated_robustness.json` (afb6e8f7ed03)<br>`results/analysis/a09_systematics.json` (4c1c137910d9) | `4883841d70db478d` |
| not included | `ch6_boom.pdf` | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_data.py` (faab74564b54)<br>`results/analysis/a06_isolated_robustness.json` (afb6e8f7ed03)<br>`results/analysis/a09_systematics.json` (4c1c137910d9) | `279b777085bf979a` |
| not included | `ch7_calib.pdf` | `ml/fig_ml.py` | `ml/results/sbi_results.json` (4ced5510c33e) | `bb5bd87cf545d0c5` |

Plotted values of the lunarmag figures: `figures/values/<file>.values.json`. Previews (PNG, 300 dpi): `figures/preview/` (Chapters 3-5), `studies/figures/`, `figures/ch6/png/`, `ml/figures/`.
