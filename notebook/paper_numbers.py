"""Every number quoted in the paper, recomputed from the stored results in results/.

    from paper_numbers import numbers, count
    numbers("4.2")      # the numbers quoted in Section 4.2, each next to the sentence it appears in
    numbers()           # all of them
    count()             # (numbers that match the paper, numbers checked)

v1.1 (thesis repository): six rows of release v1.0.0 paired a sentence with a stored value that happens to round
to the same printed number but belongs to another quantity. Two now read the right key (the spherical-voxel mean
intensity error, -3.1%, from the inversion-approach comparison; the Abstract's ~30 deg from the a03 baseline case
instead of the along-x test case), and four were moved to the sentence their value belongs to (the 3.2 deg
perfect-geometry median of the mis-pose sweep, and the 20%, 6.1 deg and 3.7 deg values of Section 4.4).
tests/test_paper_numbers.py pins all six. The count stays 374 of 374.

v1.1 also centres the skate-park mesh on the outcrop (review finding Y6-04). 21 Section 4.3 numbers move by one
rounding step or so; SUPERSEDED_Y6_04 lists each with the paper's value and the corrected one.
"""
import os, json, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def J(name):
    for sub in ("analysis", "engine"):
        path = os.path.join(ROOT, "results", sub, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(name)

# Short names for the stored results (one block per result file), then helpers that pick out the quoted values.
a2 = J('a02_isolated_outcrop_paper_geometry.json'); a3 = J('a03_three_layer_direction_vs_distance.json')
a4 = J('a04_text_number_checks.json'); bg = J('a05_background_magnetization_mc_paper_geometry.json')
cn = J('a05_correlated_noise_requirement_paper_geometry.json'); hs = J('elevated_traverse_height_sweep_results.json')
t2 = J('inversion_approach_comparison_results.json')['summary']; sp = J('skatepark_1cm_dr_sweep_results.json')['sweep']
tw = J('three_layer_wall_results.json')['arms']
b2 = J('a02b_detection_sweep_N1000.json'); a6 = J('a06_isolated_robustness.json'); a7 = J('a07_three_layer_wall_extent.json')['cases']; a8 = J('a08_skatepark_neighbor_directions.json')['results']

base = a2['baseline']; da, db = base['drive_around'], base['drive_by']
sw = {r['ratio']: r for r in b2['detection_sweep']['drive_around']}
lg = {n: {(r['pointing_deg'], r['range_m']): r['typical'] for r in g} for n, g in a2['lidar_grid'].items()}
lmax = lambda n, k: max(v[k]['median'] for v in lg[n].values())
lsub = lambda n, k: max(v[k]['median'] for (p, r), v in lg[n].items() if p <= 3 and r <= 0.15)
spA = sorted(sp['A'].values(), key=lambda r: r['delta_r_m']); spB = sorted(sp['B'].values(), key=lambda r: r['delta_r_m'])
L = lambda arm, i, k='dir_err_deg', s='mean': tw[arm]['per_layer_iterative'][f'layer_{i}'][k][s]
LS = lambda i, k='dir_err_deg', s='mean': tw['A_single_base_traverse']['per_layer_closed_form_ls'][f'layer_{i}'][k][s]
H = {h['z2_m']: h for h in hs['heights'].values()}
maxerr = hs['headline']['thresholds']['closed_form_ls']['max_dir_err_deg_vs_z2']
cu = bg['control_uniform_background']; pooled = bg['headline']['pooled_bg_to_signal']
cda = cn['isolated_outcrop']['drive_around']


LSa = lambda arm, i, k='dir_err_deg', s='mean': tw[arm]['per_layer_closed_form_ls'][f'layer_{i}'][k][s]
Bs = lambda n, grp, idx: a6['B_direction_height_scan'][n][grp][idx]
rng = lambda r, key: (min(x[key] for x in r['rows']), max(x[key] for x in r['rows']))
A1 = lambda n, m, k='dir_err_deg', s='mean': a6['A_dc_offsets'][n]['A1_iid'][m][k][s]
A2 = lambda n, m, k='dir_err_deg', s='mean': a6['A_dc_offsets'][n]['A2_offsets_5nT_rover_2nT_world'][m][k][s]
A3 = {n: {str(r['L_m']): r for r in a6['A_dc_offsets'][n]['A3_correlated']} for n in ('drive_around','drive_by')}
C = {n: {r['buried_depth_m']: r for r in a6['C_buried_extent'][n]} for n in ('drive_around','drive_by')}
D = {n: {r['kappa']: r for r in a6['D_heterogeneity'][n]} for n in ('drive_around','drive_by')}
E = {n: [r for r in a6['E_sampling_vs_ASD'][n] if r['noise']=='fixed_ASD'] for n in ('drive_around','drive_by')}
g7 = lambda c, lab, i, key='dir_err_mean': a7[c][lab][f'layer_{i}'][key]
g8 = lambda d, c, inv, k, s='median': a8[str(d)][c][inv][k][s]
hdirs = Bs('drive_around','height',1)
HH = a6['H_offsets_and_burial']; I = a6['I_survey_designs_for_offsets']
H2 = {n: {x['offset_nT']: x for x in HH[n]['H2_unfitted_offset']} for n in ('drive_around','drive_by')}
h1 = lambda n: HH[n]['H1_world_offsets_fitted']
H1r = {n: {(x['inc_deg'], x['dec_deg']): x for x in h1(n)['rows']} for n in ('drive_around','drive_by')}
H3 = {x['buried_depth_m']: x for x in HH['H3_deep_burial_drive_around']}
def sd_logu(r):
    a = 1/r; Lg = math.log(1/a); m = (1-a)/Lg; e2 = (1-a*a)/(2*Lg); return math.sqrt(e2-m*m)
_hs = sorted(hs['heights'].values(), key=lambda h: h['z2_m'])
_w = [max(h['per_layer_closed_form_ls'][f'layer_{i}']['dir_err_deg']['mean'] for i in (1,2,3)) for h in _hs]
_i = max(i for i, w in enumerate(_w) if w < 5)
z_hi = _hs[_i]['z2_m'] + (5-_w[_i])*(_hs[_i+1]['z2_m']-_hs[_i]['z2_m'])/(_w[_i+1]-_w[_i])
a9 = J('a09_systematics.json')
J9 = {(r['B_amb_nT'], r['tilt_deg']): r for r in a9['J_attitude_tilt']}
T9 = {c: {r['tilt_deg']: r for r in rows} for c, rows in a9['T_attitude_tilt_asymmetric'].items()}
K9 = {c: {(r['B0_nT'], r['R_m']): r for r in rows} for c, rows in a9['K_site_field_dipole'].items()}
DN = ('-z', '+x', 'inc45')
def kg(c, B, R, d, which='offsets_and_gradient'): return K9[c][(B, R)][d][which]['dir_err_deg']['mean']
M9 = a9['M_background_depth']
a7o = lambda c, i: a7[c]['ground+12.5m']['site_offsets_fitted'][f'layer_{i}']['dir_err_mean']
def B9(key, d, k='all_sources', m='dir_err_deg', st='mean'): return a9[key][d][k][m][st]
def rng9(keys, k='all_sources', m='dir_err_deg', st='mean', f=min): return f(B9(key, d, k, m, st) for key in keys for d in DN)
def rngb(key, f=min): return f(r['noise_free_bias']['dir_err_deg']['mean'] for dn, r in a9[key].items() if dn != 'config')
def rngk2(key):
    v=[r['all_sources']['dM_signed_pct']['mean'] for dn, r in a9[key].items() if dn != 'config']
    return f'${min(v):.0f}$--${max(v):.0f}'
def rngk(key, f=min): return f(r['all_sources']['dir_err_deg']['mean'] for dn, r in a9[key].items() if dn != 'config')
M29 = a9['M2_background_depth_two_loops']; S9 = a9['S_centring']; R9 = a9['R_burial_root_fit']; W9 = {c: {(r['B0_nT'], r['R_m']): r for r in v['rows']} for c, v in a9['W_three_layer_site_field'].items()}
N9 = {(r['B_site_nT'], r['chi_SI']): r for r in a9['N_induced_magnetization']}

SECTIONS = {
 "Abstract": "Abstract",
 "3.2": "Magnetic Field Contributions: Signals of Interest and Sources of Noise",
 "3.3": "Forward Models",
 "3.5": "Survey Geometries",
 "4.1": "Inversion Approach Selection",
 "4.2": "Isolated Outcrop",
 "4.3": "Skate Park",
 "4.4": "Three-Layer Wall",
 "5": "Discussion",
 "5.1": "Noise Requirements",
 "5.2": "Survey Design",
 "5.3": "Combined Error Budget",
 "5.5": "Limitations"
}

# (section, sentence in the paper, value printed in the paper, format, value computed from results/)
ROWS = [
 ('Abstract', '…traverse recovers the lowest flow to ∼4° but the upper two only to ∼30°; adding a traverse 6.4–13 m up the wall recovers all three to <5° if the …', '30', '{:.0f}', a3['cases']['T0_baseline']['layer_2']['dir_err_mean']),
 ('Abstract', '…we derive, with drift ≲0.01 nT rms, loops driven both ways and tilt fitted, brings it to 3–5° with loops centered to 0.1 m and burial ≤0.5 m, or with the buried …', '5', '{:.0f}', max(rngk('REQF_a_dx0.1_b0.5', f=max), rngk('REQF_c_dx0.5_b0', f=max))),
 ('3.2', '…the isolated outcrop of Section 3.3.3 produces a peak field of 0.25 nT along a traverse 1 m from its faces, and all results in Section 4 use …', '0.25', '{:.2f}', da['peak_field_nT']),
 ('3.3', "…5-m-thick slab that extends ±1.2 km laterally, which produces a field of 0.003 nT root-mean-square (rms; 0.65% of σ_B and ∼3% of the outcrop's field) along the …", '0.003', '{:.3f}', cu['uniform_field_rms_nT']),
 ('3.3', "…±1.2 km laterally, which produces a field of 0.003 nT root-mean-square (rms; 0.65% of σ_B and ∼3% of the outcrop's field) along the drive-around traverse.", '0.65', '{:.2f}', 100 * cu['ratio_to_sigma_B']),
 ('3.3', 'The slab changes the mean recovered direction error from 3.3° to 3.4° and the mean intensity error from -0.1% to -2.9% (400 inversions), so …', '3.4', '{:.1f}', cu['recovery_closed_form_ls']['dir_err_deg']['mean']),
 ('3.3', "…error from 3.3° to 3.4° and the mean intensity error from -0.1% to -2.9% (400 inversions), so even a weak coherent field (∼3% of the outcrop's) changes the …", '-2.9', '{:.1f}', cu['recovery_closed_form_ls']['mag_err_pct']['mean']),
 ('3.3', "…which produces a field of 0.003 nT root-mean-square (rms; 0.65% of σ_B and ∼3% of the outcrop's field) along the drive-around traverse.", '3', '{:.0f}', 100*cu['uniform_field_rms_nT']/da['signal_rms_per_axis_nT']),
 ('3.3', 'Their volumes are 50%, 44%, 40%, and 30% of the central outcrop, respectively, and all are magnetized at M_ref.', '44', '{:.0f}', 100*(2/3*3.14159265*1.5**3)/16),
 ('3.5', 'The drive-by survey yields ∼1600 measurements over a shorter duration.', '1600', '{:.0f}', round(db['n_stations'],-2)),
 ('4.1', 'Over 20 noise draws, the direction errors of the volume-averaged magnetization are 0.27 ± 0.13° (mean ±1-σ) for the parametric inversion, 1.24 ± 1.03° for …', '0.27', '{:.2f}', t2['parametric']['dir_err_deg_mean']),
 ('4.1', '…are 0.27 ± 0.13° (mean ±1-σ) for the parametric inversion, 1.24 ± 1.03° for the Cartesian inversion, and 3.10 ± 1.66° for the spherical …', '1.24', '{:.2f}', t2['cartesian_voxel']['dir_err_deg_mean']),
 ('4.1', '…for the parametric inversion, 1.24 ± 1.03° for the Cartesian inversion, and 3.10 ± 1.66° for the spherical inversion, and the mean intensity errors are -0.3%, -8.9%, …', '3.10', '{:.2f}', t2['spherical_voxel']['dir_err_deg_mean']),
 ('4.1', '…source geometry known, the parametric inversion recovered the magnetization direction 4.6 ± 1.0 times more accurately than the Cartesian voxel inversion and 11.6 ± 1.8 times more …', '4.6', '{:.1f}', t2['cartesian_voxel']['dir_err_deg_mean'] / t2['parametric']['dir_err_deg_mean']),
 ('4.1', '…direction 4.6 ± 1.0 times more accurately than the Cartesian voxel inversion and 11.6 ± 1.8 times more accurately than the spherical voxel inversion (Table …', '11.6', '{:.1f}', t2['spherical_voxel']['dir_err_deg_mean'] / t2['parametric']['dir_err_deg_mean']),
 ('4.1', '…the spherical inversion, and the mean intensity errors are -0.3%, -8.9%, and -3.1%, respectively.', '-3.1', '{:.1f}', t2['spherical_voxel']['mag_err_pct_mean']),
 ('4.2', "…The drive-around survey recovers the isolated outcrop's NRM direction to 3.0° and its intensity to 3.7% (medians of 1000 inversions; K = 1 parametric inversion; …", '3.0', '{:.1f}', da['dir_err_deg']['median']),
 ('4.2', "…recovers the isolated outcrop's NRM direction to 3.0° and its intensity to 3.7% (medians of 1000 inversions; K = 1 parametric inversion; Section 3.3.3), with …", '3.7', '{:.1f}', da['dM_abs_pct']['median']),
 ('4.2', '…inversions; K = 1 parametric inversion; Section 3.3.3), with a mean direction error of 3.2° and a 95th percentile of 6.1°.', '3.2', '{:.1f}', da['dir_err_deg']['mean']),
 ('4.2', '…Section 3.3.3), with a mean direction error of 3.2° and a 95th percentile of 6.1°.', '6.1', '{:.1f}', da['dir_err_deg']['p95']),
 ('4.2', 'The intensity error has almost no bias (mean signed error +0.06%) and a standard deviation of 5.5%.', '+0.06', '{:+.2f}', da['dM_signed_pct']['mean']),
 ('4.2', 'The intensity error has almost no bias (mean signed error +0.06%) and a standard deviation of 5.5%.', '5.5', '{:.1f}', da['dM_signed_pct']['std']),
 ('4.2', '…half of σ_B; the recovery instead relies on fitting three unknowns to 7197 data (2399 three-component measurements), over which the noise averages down.', '2399', '{}', da['n_stations']),
 ('4.2', 'The drive-by survey recovers the direction to 6.6° and the intensity to 7.3% (medians; mean direction error 7.3°), with …', '6.6', '{:.1f}', db['dir_err_deg']['median']),
 ('4.2', '…survey recovers the direction to 6.6° and the intensity to 7.3% (medians; mean direction error 7.3°), with the largest error in the along-track component, M_x, whose …', '7.3', '{:.1f}', db['dir_err_deg']['mean']),
 ('4.2', 'The drive-by survey recovers the direction to 6.6° and the intensity to 7.3% (medians; mean direction error 7.3°), with the largest error in the along-track component, …', '7.3', '{:.1f}', db['dM_abs_pct']['median']),
 ('4.2', '…with the largest error in the along-track component, M_x, whose scatter is 3.1 times larger than for the drive-around because a single pass on one side of the outcrop does not measure …', '3.1', '{:.1f}', a4['drive_by_over_drive_around_std']['x']),
 ('4.2', 'The median errors are 6.0° and 6.7% at half the reference magnetization, 1.5° and 1.8% at twice the …', '6.0', '{:.1f}', sw[0.5]['dir_err_deg']['median']),
 ('4.2', 'The median errors are 6.0° and 6.7% at half the reference magnetization, 1.5° and 1.8% at twice the reference, and …', '6.7', '{:.1f}', sw[0.5]['dM_abs_pct']['median']),
 ('4.2', 'The median errors are 6.0° and 6.7% at half the reference magnetization, 1.5° and 1.8% at twice the reference, and 0.3° and 0.3% at 10 times the reference, …', '1.5', '{:.1f}', sw[2]['dir_err_deg']['median']),
 ('4.2', '…are 6.0° and 6.7% at half the reference magnetization, 1.5° and 1.8% at twice the reference, and 0.3° and 0.3% at 10 times the reference, and at one-tenth of …', '1.8', '{:.1f}', sw[2]['dM_abs_pct']['median']),
 ('4.2', '…half the reference magnetization, 1.5° and 1.8% at twice the reference, and 0.3° and 0.3% at 10 times the reference, and at one-tenth of the reference the median direction …', '0.3', '{:.1f}', sw[10]['dir_err_deg']['median']),
 ('4.2', '…magnetization, 1.5° and 1.8% at twice the reference, and 0.3° and 0.3% at 10 times the reference, and at one-tenth of the reference the median direction error reaches …', '0.3', '{:.1f}', sw[10]['dM_abs_pct']['median']),
 ('4.2', '…10 times the reference, and at one-tenth of the reference the median direction error reaches 30°.', '30', '{:.0f}', sw[0.1]['dir_err_deg']['median']),
 ('4.2', 'The mean direction error over all directions is 3.2–3.6° for the drive-around survey, with horizontal magnetizations …', '3.2', '{:.1f}', hdirs['dir_err_mean_min']),
 ('4.2', 'The mean direction error over all directions is 3.2–3.6° for the drive-around survey, with horizontal magnetizations slightly worse than …', '3.6', '{:.1f}', hdirs['dir_err_mean_max']),
 ('4.2', '…survey, with horizontal magnetizations slightly worse than vertical ones, and 6.4–8.6° for the drive-by survey.', '6.4', '{:.1f}', Bs('drive_by','height',1)['dir_err_mean_min']),
 ('4.2', '…survey, with horizontal magnetizations slightly worse than vertical ones, and 6.4–8.6° for the drive-by survey.', '8.6', '{:.1f}', Bs('drive_by','height',1)['dir_err_mean_max']),
 ('4.2', '…computed the estimator covariance, σ_B^2(G_K^TG_K)⁻¹, for 42 magnetization directions spanning all inclinations and declinations and drew 4000 recoveries from it for …', '42', '{}', len(hdirs['rows'])),
 ('4.2', 'Sensor heights from 0.25 to 1.0 m give 3.2–3.8°, and raising the sensor to 1.5 m, above the top of the outcrop, gives …', '3.2', '{:.1f}', min(Bs('drive_around','height',i)['dir_err_mean_min'] for i in (0,1,2,3))),
 ('4.2', 'Sensor heights from 0.25 to 1.0 m give 3.2–3.8°, and raising the sensor to 1.5 m, above the top of the outcrop, gives …', '3.8', '{:.1f}', max(Bs('drive_around','height',i)['dir_err_mean_max'] for i in (0,1,2,3))),
 ('4.2', '…3.2–3.8°, and raising the sensor to 1.5 m, above the top of the outcrop, gives 4.5–4.6°.', '4.5', '{:.1f}', Bs('drive_around','height',4)['dir_err_mean_min']),
 ('4.2', '…and raising the sensor to 1.5 m, above the top of the outcrop, gives 4.5–4.6°.', '4.6', '{:.1f}', Bs('drive_around','height',4)['dir_err_mean_max']),
 ('4.2', 'The standoff matters more than the height; halving it to 0.5 m roughly halves the error (1.7–1.8°), and doubling it to 2 m raises the error to 7.7–9.1°.', '1.7', '{:.1f}', Bs('drive_around','standoff',0)['dir_err_mean_min']),
 ('4.2', 'The standoff matters more than the height; halving it to 0.5 m roughly halves the error (1.7–1.8°), and doubling it to 2 m raises the error to 7.7–9.1°.', '1.8', '{:.1f}', Bs('drive_around','standoff',0)['dir_err_mean_max']),
 ('4.2', '…halves the error (1.7–1.8°), and doubling it to 2 m raises the error to 7.7–9.1°.', '7.7', '{:.1f}', Bs('drive_around','standoff',2)['dir_err_mean_min']),
 ('4.2', '…halves the error (1.7–1.8°), and doubling it to 2 m raises the error to 7.7–9.1°.', '9.1', '{:.1f}', Bs('drive_around','standoff',2)['dir_err_mean_max']),
 ('4.2', 'Figure 5 caption: Over the full grid, the drive-around survey recovers the direction to ≤5.4° and the intensity to ≤10.6%, and the drive-by survey to ≤11.3° …', '5.4', '{:.1f}', lmax('drive_around', 'dir_err_deg')),
 ('4.2', '…the imaged outcrop position raise the drive-around errors to at most 5.4° and 10.6% (medians) over the full range we tested (Figure 5).', '10.6', '{:.1f}', lmax('drive_around', 'dM_abs_pct')),
 ('4.2', '…in this sweep is 3.2°, and within 3° and 0.15 m the medians are ≤3.8° and ≤4.6%.', '3.8', '{:.1f}', lsub('drive_around', 'dir_err_deg')),
 ('4.2', '…and within 3° and 0.15 m the medians are ≤3.8° and ≤4.6%.', '4.6', '{:.1f}', lsub('drive_around', 'dM_abs_pct')),
 ('4.2', 'The drive-by survey is more sensitive, with medians up to 11.3° and 20.8%.', '11.3', '{:.1f}', lmax('drive_by', 'dir_err_deg')),
 ('4.2', 'The drive-by survey is more sensitive, with medians up to 11.3° and 20.8%.', '20.8', '{:.1f}', lmax('drive_by', 'dM_abs_pct')),
 ('4.2', '…where a 0.30 m range bound is combined with a pointing bound of 4–6° (5.4–5.6° from 100 draws per cell, i.e., close to the goal).', '5.4', '{:.1f}', min(v['dir_err_deg']['mean'] for k,v in lg['drive_around'].items() if v['dir_err_deg']['mean']>5)),
 ('4.2', '…where a 0.30 m range bound is combined with a pointing bound of 4–6° (5.4–5.6° from 100 draws per cell, i.e., close to the goal).', '5.6', '{:.1f}', max(v['dir_err_deg']['mean'] for k,v in lg['drive_around'].items() if v['dir_err_deg']['mean']>5)),
 ('4.2', 'With perfect geometry the median direction error in this sweep is 3.2°, and within 3° and 0.15 m the medians are ≤3.8° and ≤4.6%.', '3.2', '{:.1f}', lg['drive_around'][(0.0, 0.0)]['dir_err_deg']['median']),
 ('4.2', '…mainly affect the direction; a 6° pointing error alone raises the median to 4.1° and leaves the intensity error at 3.1%.', '4.1', '{:.1f}', lg['drive_around'][(6.0, 0.0)]['dir_err_deg']['median']),
 ('4.2', '…error alone raises the median to 4.1° and leaves the intensity error at 3.1%.', '3.1', '{:.1f}', lg['drive_around'][(6.0, 0.0)]['dM_abs_pct']['median']),
 ('4.2', '…a traverse only 1 m away; a 0.30 m range error alone raises the intensity error to 8.1%, with a direction error of 3.5°.', '8.1', '{:.1f}', lg['drive_around'][(0.0, 0.3)]['dM_abs_pct']['median']),
 ('4.2', '…m range error alone raises the intensity error to 8.1%, with a direction error of 3.5°.', '3.5', '{:.1f}', lg['drive_around'][(0.0, 0.3)]['dir_err_deg']['median']),
 ('4.2', 'For the drive-by survey, range errors degrade both (to 10.6° and 19.0% at 0.30 m).', '10.6', '{:.1f}', lg['drive_by'][(0.0, 0.3)]['dir_err_deg']['median']),
 ('4.2', 'For the drive-by survey, range errors degrade both (to 10.6° and 19.0% at 0.30 m).', '19.0', '{:.1f}', lg['drive_by'][(0.0, 0.3)]['dM_abs_pct']['median']),
 ('4.2', 'The intensity is biased high, by +16–20% at b = 0.25 m and +26–35% at b = 0.5 m, because the model assigns the field of the …', '+16', '{:+.0f}', C['drive_around'][0.25]['-z']['dM_signed_pct']['mean']),
 ('4.2', 'The intensity is biased high, by +16–20% at b = 0.25 m and +26–35% at b = 0.5 m, because the model assigns the field of the …', '20', '{:.0f}', C['drive_around'][0.25]['+x']['dM_signed_pct']['mean']),
 ('4.2', 'The intensity is biased high, by +16–20% at b = 0.25 m and +26–35% at b = 0.5 m, because the model assigns the field of the buried part to the exposed part.', '+26', '{:+.0f}', C['drive_around'][0.5]['-z']['dM_signed_pct']['mean']),
 ('4.2', 'The intensity is biased high, by +16–20% at b = 0.25 m and +26–35% at b = 0.5 m, because the model assigns the field of the buried part to the exposed part.', '35', '{:.0f}', C['drive_around'][0.5]['+x']['dM_signed_pct']['mean']),
 ('4.2', 'For a magnetization inclined at 45°, the noise-free bias is 1.8° at b = 0.5 m and 4.2° at b = 1 m.', '1.8', '{:.1f}', C['drive_around'][0.5]['inc45']['noise_free_dir_err_deg']),
 ('4.2', '…part from one side only, its direction error for a vertical magnetization grows to 8.4° at b = 0.25 m and 12.4° at b = 0.5 m.', '8.4', '{:.1f}', C['drive_by'][0.25]['-z']['dir_err_deg']['mean']),
 ('4.2', '…error for a vertical magnetization grows to 8.4° at b = 0.25 m and 12.4° at b = 0.5 m.', '12.4', '{:.1f}', C['drive_by'][0.5]['-z']['dir_err_deg']['mean']),
 ('4.2', '…survey and vertical or horizontal magnetizations, the direction error stays at 2.3–3.2° (mean) for 0 < b ≤ 1 m, falling slightly with b only because the inflated …', '2.3', '{:.1f}', min(C['drive_around'][b][d]['dir_err_deg']['mean'] for b in (0.1,0.25,0.5,1.0) for d in ('-z','+x'))),
 ('4.2', '…survey and vertical or horizontal magnetizations, the direction error stays at 2.3–3.2° (mean) for 0 < b ≤ 1 m, falling slightly with b only because the inflated …', '3.2', '{:.1f}', max(C['drive_around'][b][d]['dir_err_deg']['mean'] for b in (0.1,0.25,0.5,1.0) for d in ('-z','+x'))),
 ('4.2', '…Fisher mean (κ = 10), the drive-around recovers the volume-mean vector with a noise-free bias of 1.2°, and the direction error relative to the Fisher mean is 5.6°.', '1.2', '{:.1f}', D['drive_around'][10.0]['noise_free_err_vs_volume_mean']['mean']),
 ('4.2', '…vector with a noise-free bias of 1.2°, and the direction error relative to the Fisher mean is 5.6°.', '5.6', '{:.1f}', D['drive_around'][10.0]['err_vs_fisher_mean']['mean']),
 ('4.2', 'For cells whose directions lie a mean of 23° from the Fisher mean (κ = 10), the drive-around recovers the volume-mean …', '23', '{:.0f}', D['drive_around'][10.0]['mean_cell_angle_from_mean_deg']),
 ('4.2', 'For a mean angle of 33° (κ = 5), these become 2.0° and 7.7°, respectively.', '2.0', '{:.1f}', D['drive_around'][5.0]['noise_free_err_vs_volume_mean']['mean']),
 ('4.2', 'For a mean angle of 33° (κ = 5), these become 2.0° and 7.7°, respectively.', '7.7', '{:.1f}', D['drive_around'][5.0]['err_vs_fisher_mean']['mean']),
 ('4.2', '…whereas the drive-by weights the near side strongly and its noise-free bias reaches 6.3° and 9.7° for the same two cases.', '6.3', '{:.1f}', D['drive_by'][10.0]['noise_free_err_vs_volume_mean']['mean']),
 ('4.2', '…weights the near side strongly and its noise-free bias reaches 6.3° and 9.7° for the same two cases.', '9.7', '{:.1f}', D['drive_by'][5.0]['noise_free_err_vs_volume_mean']['mean']),
 ('4.2', '…inclined at 45°, the noise-free bias is 1.8° at b = 0.5 m and 4.2° at b = 1 m.', '4.2', '{:.1f}', C['drive_around'][1.0]['inc45']['noise_free_dir_err_deg']),
 ('4.2', 'Left unfitted, a constant offset of 0.03 nT raises the drive-around error to 3.9° and one of 0.1 nT to 9.0°.', '3.9', '{:.1f}', H2['drive_around'][0.03]['dir_err_deg']['mean']),
 ('4.2', '…a constant offset of 0.03 nT raises the drive-around error to 3.9° and one of 0.1 nT to 9.0°.', '9.0', '{:.1f}', H2['drive_around'][0.1]['dir_err_deg']['mean']),
 ('4.2', 'Table 3: Unfitted constant offset, 0.1 nT | 9.0 | 22.7 | 13.2', '22.7', '{:.1f}', H2['drive_around'][0.1]['dM_abs_pct']['median']),
 ('4.2', 'Table 3: Unfitted constant offset, 0.1 nT | 9.0 | 22.7 | 13.2', '13.2', '{:.1f}', H2['drive_by'][0.1]['dir_err_deg']['mean']),
 ('4.2', 'Fitting three local-frame offsets on a single loop leaves the direction error of a vertical magnetization at 3.7° but raises the intensity error to 13% and the direction …', '3.7', '{:.1f}', H1r['drive_around'][(-90,0)]['dir_err_mean']),
 ('4.2', '…leaves the direction error of a vertical magnetization at 3.7° but raises the intensity error to 13% and the direction error of horizontal magnetizations to 9.1°, because …', '13', '{:.0f}', H1r['drive_around'][(-90,0)]['dM_abs_median']),
 ('4.2', '…at 3.7° but raises the intensity error to 13% and the direction error of horizontal magnetizations to 9.1°, because along a single loop a constant vertical field closely …', '9.1', '{:.1f}', h1('drive_around')['dir_err_mean_max']),
 ('4.2', 'Table 3: Local offsets fitted, one loop (all directions) | 3.6–9.1 | 3.6–12.8 | 7.8–9.9', '3.6', '{:.1f}', h1('drive_around')['dir_err_mean_min']),
 ('4.2', 'Table 3: Local offsets fitted, one loop (all directions) | 3.6–9.1 | 3.6–12.8 | 7.8–9.9', '12.8', '{:.1f}', max(x['dM_abs_median'] for x in h1('drive_around')['rows'])),
 ('4.2', 'Table 3: Local offsets fitted, one loop (all directions) | 3.6–9.1 | 3.6–12.8 | 7.8–9.9', '3.6', '{:.1f}', min(x['dM_abs_median'] for x in h1('drive_around')['rows'])),
 ('4.2', 'Table 3: Local offsets fitted, one loop (all directions) | 3.6–9.1 | 3.6–12.8 | 7.8–9.9', '7.8', '{:.1f}', h1('drive_by')['dir_err_mean_min']),
 ('4.2', 'Table 3: Local offsets fitted, one loop (all directions) | 3.6–9.1 | 3.6–12.8 | 7.8–9.9', '9.9', '{:.1f}', h1('drive_by')['dir_err_mean_max']),
 ('4.2', 'A 10 m straight approach leg brings the range over all 42 directions to 3.4–5.4°, and a second loop at 3 m standoff brings it to 3.3–4.4° with intensity …', '3.4', '{:.1f}', I['loop_1m+10m_leg']['local_offsets']['dir_err_mean_min']),
 ('4.2', 'A 10 m straight approach leg brings the range over all 42 directions to 3.4–5.4°, and a second loop at 3 m standoff brings it to 3.3–4.4° with intensity …', '5.4', '{:.1f}', I['loop_1m+10m_leg']['local_offsets']['dir_err_mean_max']),
 ('4.2', '…over all 42 directions to 3.4–5.4°, and a second loop at 3 m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when …', '3.3', '{:.1f}', I['loop_1m+loop_3m']['local_offsets']['dir_err_mean_min']),
 ('4.2', '…42 directions to 3.4–5.4°, and a second loop at 3 m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when a uniform …', '4.4', '{:.1f}', I['loop_1m+loop_3m']['local_offsets']['dir_err_mean_max']),
 ('4.2', '…and a second loop at 3 m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when a uniform field gradient is fitted as …', '3.1', '{:.1f}', min(x['dM_abs_median'] for x in I['loop_1m+loop_3m']['local_offsets']['rows'])),
 ('4.2', '…second loop at 3 m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when a uniform field gradient is fitted as well.', '5.0', '{:.1f}', max(x['dM_abs_median'] for x in I['loop_1m+loop_3m']['local_offsets']['rows'])),
 ('4.2', '…m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when a uniform field gradient is fitted as well.', '3.3', '{:.1f}', I['loop_1m+loop_3m']['local_offsets_and_gradient']['dir_err_mean_min']),
 ('4.2', '…m standoff brings it to 3.3–4.4° with intensity errors of 3.1–5.0%, and to 3.3–4.5° when a uniform field gradient is fitted as well.', '4.5', '{:.1f}', I['loop_1m+loop_3m']['local_offsets_and_gradient']['dir_err_mean_max']),
 ('4.2', 'Table 3: Offsets and gradient fitted, loops at 1 and 3 m | 3.3–4.5 | 3.1–5.1', '5.1', '{:.1f}', max(x['dM_abs_median'] for x in I['loop_1m+loop_3m']['local_offsets_and_gradient']['rows'])),
 ('4.2', '…an inclined magnetization more; in noise-free tests, a 45° magnetization is off by 9.1° at b = 2 m, 13.3° at b = 3 m, and 19.1° at b = 5 m, while …', '9.1', '{:.1f}', H3[2.0]['inc45']['noise_free_dir_err_deg']),
 ('4.2', '…noise-free tests, a 45° magnetization is off by 9.1° at b = 2 m, 13.3° at b = 3 m, and 19.1° at b = 5 m, while vertical and horizontal magnetizations …', '13.3', '{:.1f}', H3[3.0]['inc45']['noise_free_dir_err_deg']),
 ('4.2', '…figure Errors in the imaged outcrop position raise the drive-around errors to at most 5.4° and 10.6% (medians) over the full range we tested (Figure …', '10.6', '{:.1f}', lmax('drive_around','dM_abs_pct')),
 ('4.2', 'The largest intensity error occurs at 1° and 0.30 m, with the corner giving 8.6%.', '8.6', '{:.1f}', lg['drive_around'][(6.0,0.3)]['dM_abs_pct']['median']),
 ('4.2', '…gradient does not describe; when that part is ≲0.02 nT rms along the loops, the direction error is 2.6–4.9° in every case we ran (sensor heights of 0.25–1 m, and loops …', '2.6', '{:.1f}', min(kg(c,B,R,d) for c in K9 for (B,R) in ((30.,300.),(30.,1000.),(100.,1000.),(300.,1000.)) for d in DN)),
 ('4.2', '…when that part is ≲0.02 nT rms along the loops, the direction error is 2.6–4.9° in every case we ran (sensor heights of 0.25–1 m, and loops 0.5 m off center).', '4.9', '{:.1f}', max(kg(c,B,R,d) for c in K9 for (B,R) in ((30.,300.),(30.,1000.),(100.,1000.),(300.,1000.)) for d in DN)),
 ('4.2', '…the part of the site field that a uniform gradient does not describe; when that part is ≲0.02 nT rms along the loops, the direction error is 2.6–4.9° in every case we ran …', '0.02', '{:.2f}', max(K9[c][(B,R)]['residual_after_gradient_rms_nT'] for c in K9 for (B,R) in ((30.,300.),(30.,1000.),(100.,1000.),(300.,1000.)))),
 ('4.2', '…from ≳1 km meets the 0.02 nT limit, whereas a 100 nT field from 300 m (0.06 nT left over) gives 3.3–8.6° and a 300 nT field from 300 m gives up to 20°.', '0.06', '{:.2f}', K9['h0.5'][(100.,300.)]['residual_after_gradient_rms_nT']),
 ('4.2', '…km meets the 0.02 nT limit, whereas a 100 nT field from 300 m (0.06 nT left over) gives 3.3–8.6° and a 300 nT field from 300 m gives up to 20°.', '3.3', '{:.1f}', min(kg(c,100.,300.,d) for c in K9 for d in DN)),
 ('4.2', '…the 0.02 nT limit, whereas a 100 nT field from 300 m (0.06 nT left over) gives 3.3–8.6° and a 300 nT field from 300 m gives up to 20°.', '8.6', '{:.1f}', max(kg(c,100.,300.,d) for c in K9 for d in DN)),
 ('4.2', '…300 m (0.06 nT left over) gives 3.3–8.6° and a 300 nT field from 300 m gives up to 20°.', '20', '{:.0f}', max(kg(c,300.,300.,d) for c in K9 for d in DN)),
 ('4.2', 'Without the gradient terms, off-center loops or other sensor heights give 7–78° even for sources 1 km away.', '7', '{:.0f}', min(kg(c,B,1000.,d,'offsets_only') for c in ('h0.25','h1.0','offcentre0.5m') for B in (30.,100.,300.) for d in DN)),
 ('4.2', 'Without the gradient terms, off-center loops or other sensor heights give 7–78° even for sources 1 km away.', '78', '{:.0f}', max(kg(c,B,1000.,d,'offsets_only') for c in ('h0.25','h1.0','offcentre0.5m') for B in (30.,100.,300.) for d in DN)),
 ('4.2', 'Table 3: Standoff 0.5 m (all directions) | 1.7–1.8 | 1.6–1.8 | 3.1–4.6', '3.1', '{:.1f}', a6['B_direction_height_scan']['drive_by']['standoff'][0]['dir_err_mean_min']),
 ('4.2', '…magnetization is off by 9.1° at b = 2 m, 13.3° at b = 3 m, and 19.1° at b = 5 m, while vertical and horizontal magnetizations stay unbiased in direction.', '19.1', '{:.1f}', HH['H3_deep_burial_drive_around'][3]['inc45']['noise_free_dir_err_deg']),
 ('4.2', '…for b of 1–3 m, although for a vertical magnetization it shrinks with depth and turns to -11% at b = 5 m, which implies that for deep burial the recovered intensity is not an upper …', '-11', '{:.0f}', HH['H3_deep_burial_drive_around'][3]['-z']['noise_free_dM_pct']),
 ('4.2', 'The intensity bias is +11 to +80% for b of 1–3 m, although for a vertical magnetization it shrinks with depth and turns …', '80', '{:.0f}', max(x[d]['noise_free_dM_pct'] for x in HH['H3_deep_burial_drive_around'][:3] for d in DN)),
 ('4.2', 'The intensity bias is +11 to +80% for b of 1–3 m, although for a vertical magnetization it shrinks with depth and turns …', '11', '{:.0f}', min(x[d]['noise_free_dM_pct'] for x in HH['H3_deep_burial_drive_around'][:3] for d in DN)),
 ('4.2', 'Table 3: Buried extent 1 m, 45° inclination | 4.2 (bias) | +44 (bias)', '44', '{:.0f}', HH['H3_deep_burial_drive_around'][0]['inc45']['noise_free_dM_pct']),
 ('4.2', 'Table 3: Buried extent 3 m, 45° inclination | 13.3 (bias) | +50 (bias)', '50', '{:.0f}', HH['H3_deep_burial_drive_around'][2]['inc45']['noise_free_dM_pct']),
 ('4.2', '…2°, although this is approximate because the errors are not quite isotropic (the vertical scatter is 1.3 times the horizontal for the drive-around).', '1.3', '{:.1f}', a4['drive_around_component_std_Am']['z']/a4['drive_around_component_std_Am']['x']),
 ('4.2', '…extent drawn between 0 and 1 m, moving both loops off center by 0.1, 0.25, and 0.5 m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° …', '2.3', '{:.1f}', min(S9['dx0.1'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…both loops off center by 0.1, 0.25, and 0.5 m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° over the three magnetization …', '5.3', '{:.1f}', max(S9['dx0.1'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…by 0.1, 0.25, and 0.5 m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° over the three magnetization directions we tested (60 draws …', '5.0', '{:.1f}', min(S9['dx0.25'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…by 0.1, 0.25, and 0.5 m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° over the three magnetization directions we tested (60 draws …', '9.0', '{:.1f}', max(S9['dx0.25'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…0.5 m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° over the three magnetization directions we tested (60 draws each; standard errors …', '9.3', '{:.1f}', min(S9['dx0.5'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…m gives noise-free direction biases of 2.3–5.3°, 5.0–9.0°, and 9.3–13.3° over the three magnetization directions we tested (60 draws each; standard errors …', '13.3', '{:.1f}', max(S9['dx0.5'][d]['burial_bias_deg']['mean'] for d in DN)),
 ('4.2', '…from the pose error tested above (± 3°, ± 0.15 m) grows the same way, from a bias of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and …', '1.2', '{:.1f}', min(S9['dx0'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…error tested above (± 3°, ± 0.15 m) grows the same way, from a bias of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and 4.5–7.3° at …', '1.6', '{:.1f}', max(S9['dx0'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…m) grows the same way, from a bias of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and 4.5–7.3° at 0.5 m off center.', '3.0', '{:.1f}', min(S9['dx0.25'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…m) grows the same way, from a bias of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and 4.5–7.3° at 0.5 m off center.', '4.0', '{:.1f}', max(S9['dx0.25'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…bias of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and 4.5–7.3° at 0.5 m off center.', '4.5', '{:.1f}', min(S9['dx0.5'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…of 1.2–1.6° with the loops centered to 3.0–4.0° at 0.25 m and 4.5–7.3° at 0.5 m off center.', '7.3', '{:.1f}', max(S9['dx0.5'][d]['pose_bias_deg']['mean'] for d in DN)),
 ('4.2', '…magnetization) removes most of the burial bias; with the loops 0.5 m off center, it leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of …', '0.6', '{:.1f}', min(R9[f'dx0.5_bmax1_{d}']['with_root_1m']['noise_free']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…removes most of the burial bias; with the loops 0.5 m off center, it leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of 8.9–14.5°.', '1.4', '{:.1f}', max(R9[f'dx0.5_bmax1_{d}']['with_root_1m']['noise_free']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…bias; with the loops 0.5 m off center, it leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of 8.9–14.5°.', '4.7', '{:.1f}', min(R9[f'dx0.5_bmax1_{d}']['with_root_1m']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…the loops 0.5 m off center, it leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of 8.9–14.5°.', '5.3', '{:.1f}', max(R9[f'dx0.5_bmax1_{d}']['with_root_1m']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…center, it leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of 8.9–14.5°.', '8.9', '{:.1f}', min(R9[f'dx0.5_bmax1_{d}']['exposed_only']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…leaves 0.6–1.4° of bias and mean errors of 4.7–5.3° instead of 8.9–14.5°.', '14.5', '{:.1f}', max(R9[f'dx0.5_bmax1_{d}']['exposed_only']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', 'With the loops centered, the extra unknowns cost precision (5.2–5.4° instead of 2.9–4.3°), so the root is worth fitting only when the loops …', '5.2', '{:.1f}', min(R9[f'dx0_bmax1_{d}']['with_root_1m']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', 'With the loops centered, the extra unknowns cost precision (5.2–5.4° instead of 2.9–4.3°), so the root is worth fitting only when the loops …', '5.4', '{:.1f}', max(R9[f'dx0_bmax1_{d}']['with_root_1m']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', 'With the loops centered, the extra unknowns cost precision (5.2–5.4° instead of 2.9–4.3°), so the root is worth fitting only when the loops cannot be centered or the …', '2.9', '{:.1f}', min(R9[f'dx0_bmax1_{d}']['exposed_only']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.2', '…the loops centered, the extra unknowns cost precision (5.2–5.4° instead of 2.9–4.3°), so the root is worth fitting only when the loops cannot be centered or the root is …', '4.3', '{:.1f}', max(R9[f'dx0_bmax1_{d}']['exposed_only']['all']['dir_err_deg']['mean'] for d in DN)),
 ('4.3', '…(configuration A), the intensity is biased high, with a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+45', '{:+.0f}', spA[0]['mag_err_pct_signed']['mean']),
 ('4.3', '…is biased high, with a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+20', '{:+.0f}', spA[1]['mag_err_pct_signed']['mean']),
 ('4.3', '…a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+11', '{:+.0f}', spA[2]['mag_err_pct_signed']['mean']),
 ('4.3', 'The median direction error is 2.4–3.0° at every Δ r and is slightly lower than the isolated value at small Δ r …', '2.4', '{:.1f}', min(r['dir_err_deg']['median'] for r in spA)),
 ('4.3', 'The median direction error is 2.4–3.0° at every Δ r and is slightly lower than the isolated value at small Δ r …', '3.0', '{:.1f}', max(r['dir_err_deg']['median'] for r in spA)),
 ('4.3', 'For Δ r 5 m, the median intensity error is ≤5.3%, and at 8–10 m it is 3.3–3.7%, comparable to the isolated value of 3.7%.', '5.3', '{:.1f}', max(r['mag_err_pct_abs']['median'] for r in spA if r['delta_r_m'] >= 5)),
 ('4.3', 'The residual RMS is 1.002 at Δ r = 1 m and 1.00 elsewhere, within the 0.008 scatter expected for 7197 …', '1.002', '{:.3f}', spA[0]['residual_rms']['median']),
 ('4.3', 'The median direction error is 4.4° at Δ r = 1 m and 2.9–3.2° from 2 to 10 m, which indicates that neighbors …', '4.4', '{:.1f}', spB[0]['dir_err_deg']['median']),
 ('4.3', 'The median direction error is 4.4° at Δ r = 1 m and 2.9–3.2° from 2 to 10 m, which indicates that neighbors that record the opposite polarity do …', '3.2', '{:.1f}', max(r['dir_err_deg']['median'] for r in spB[1:])),
 ('4.3', '…(configuration B), the intensity bias disappears, with a median intensity error of 3.0–4.4% and a mean signed error between -3.1% and +0.2% at every Δ r, because the …', '4.4', '{:.1f}', max(r['mag_err_pct_abs']['median'] for r in spB)),
 ('4.3', '…all four neighbors magnetized along +x or +y at Δ r = 1 m, the median direction error is 10.1° and 6.9°, respectively, and with each neighbor in an …', '10.1', '{:.1f}', g8(1,'HX','central','dir_err_deg')),
 ('4.3', '…along +x or +y at Δ r = 1 m, the median direction error is 10.1° and 6.9°, respectively, and with each neighbor in an independent random direction …', '6.9', '{:.1f}', g8(1,'HY','central','dir_err_deg')),
 ('4.3', '…6.9°, respectively, and with each neighbor in an independent random direction the median is 6.8° and the 95th percentile 13.8°.', '6.8', '{:.1f}', g8(1,'RND','central','dir_err_deg')),
 ('4.3', '…each neighbor in an independent random direction the median is 6.8° and the 95th percentile 13.8°.', '13.8', '{:.1f}', g8(1,'RND','central','dir_err_deg','p95')),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '3.7', '{:.1f}', min(g8(2,c,'central','dir_err_deg') for c in ('HX','HY','RND'))),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '4.6', '{:.1f}', max(g8(2,c,'central','dir_err_deg') for c in ('HX','HY','RND'))),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '2.8', '{:.1f}', min(g8(5,c,'central','dir_err_deg') for c in ('HX','HY','RND'))),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '3.2', '{:.1f}', max(g8(5,c,'central','dir_err_deg') for c in ('HX','HY','RND'))),
 ('4.3', '…the inversion (K = 5) removes most of the bias, with a median direction error of the central outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means …', '3.1', '{:.1f}', min(g8(1,c,'joint','dir_err_deg') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…= 5) removes most of the bias, with a median direction error of the central outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means 5.1–5.8°, just above the …', '4.1', '{:.1f}', max(g8(1,c,'joint','dir_err_deg') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…(means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.3', '{:.1f}', max(abs(g8(1,c,'joint','dM_signed_pct')) for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…fields are partly correlated along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '23', '{:.0f}', min(g8(1,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…fields are partly correlated along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '29', '{:.0f}', max(g8(1,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '16', '{:.0f}', min(g8(2,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '18', '{:.0f}', max(g8(2,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '12', '{:.0f}', min(g8(5,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '13', '{:.0f}', max(g8(5,c,'joint','dM_abs_pct') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…the central outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.1', '{:.1f}', min(g8(1,c,'joint','dir_err_deg','mean') for c in ('A','B','HX','HY','RND'))),
 ('4.3', '…outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.8', '{:.1f}', max(g8(1,c,'joint','dir_err_deg','mean') for c in ('A','B','HX','HY','RND'))),
 ('4.3', 'At Δ r = 2 m the means are 3.9–4.3°.', '3.9', '{:.1f}', min(g8(2,c,'joint','dir_err_deg','mean') for c in ('A','B','HX','HY','RND'))),
 ('4.3', 'At Δ r = 2 m the means are 3.9–4.3°.', '4.3', '{:.1f}', max(g8(2,c,'joint','dir_err_deg','mean') for c in ('A','B','HX','HY','RND'))),
 ('4.4', '…traverse in front of the wall (|x| ≤ 50 m), the rms fields of the three flows are 0.58, 0.19, and 0.11 nT, compared with σ_B = 0.50 nT.', '0.58', '{:.2f}', a7['W50']['ground']['signal_rms_nT_absx_le_50'][0]),
 ('4.4', '…in front of the wall (|x| ≤ 50 m), the rms fields of the three flows are 0.58, 0.19, and 0.11 nT, compared with σ_B = 0.50 nT.', '0.19', '{:.2f}', a7['W50']['ground']['signal_rms_nT_absx_le_50'][1]),
 ('4.4', '…of the wall (|x| ≤ 50 m), the rms fields of the three flows are 0.58, 0.19, and 0.11 nT, compared with σ_B = 0.50 nT.', '0.11', '{:.2f}', a7['W50']['ground']['signal_rms_nT_absx_le_50'][2]),
 ('4.4', 'Across 1000 inversions, the undamped solution recovers the lowest flow to 3.7 ± 2.0° (mean ±1-σ) and the middle and highest flows only to 28 ± 20° …', '3.7', '{:.1f}', LSa('A_single_base_traverse', 1)),
 ('4.4', '…3.7 ± 2.0° (mean ±1-σ) and the middle and highest flows only to 28 ± 20° and 33 ± 20°, with mean intensities 18% and 29% too high because noise …', '28', '{:.0f}', LSa('A_single_base_traverse', 2)),
 ('4.4', '…±1-σ) and the middle and highest flows only to 28 ± 20° and 33 ± 20°, with mean intensities 18% and 29% too high because noise inflates the magnitude …', '33', '{:.0f}', LSa('A_single_base_traverse', 3)),
 ('4.4', '…middle and highest flows only to 28 ± 20° and 33 ± 20°, with mean intensities 18% and 29% too high because noise inflates the magnitude of poorly constrained vectors; …', '18', '{:.0f}', LSa('A_single_base_traverse', 2, 'mag_err_pct')),
 ('4.4', '…only to 28 ± 20° and 33 ± 20°, with mean intensities 18% and 29% too high because noise inflates the magnitude of poorly constrained vectors; these errors are too …', '29', '{:.0f}', LSa('A_single_base_traverse', 3, 'mag_err_pct')),
 ('4.4', 'The damped solution narrows the scatter of the two upper flows to 24 ± 8° and 34 ± 10° by shrinking them toward zero, which biases their intensities low …', '24', '{:.0f}', L('A_single_base_traverse', 2)),
 ('4.4', 'The damped solution narrows the scatter of the two upper flows to 24 ± 8° and 34 ± 10° by shrinking them toward zero, which biases their intensities low by 17% and 20% …', '34', '{:.0f}', L('A_single_base_traverse', 3)),
 ('4.4', '…34 ± 10° by shrinking them toward zero, which biases their intensities low by 17% and 20% (Table 4), although the exact upper-flow errors depend on …', '17', '{:.0f}', -L('A_single_base_traverse', 2, 'mag_err_pct')),
 ('4.4', '…34 ± 10° by shrinking them toward zero, which biases their intensities low by 17% and 20% (Table 4), although the exact upper-flow errors depend on …', '20', '{:.0f}', -L('A_single_base_traverse', 3, 'mag_err_pct')),
 ('4.4', 'Table 4: 2 (middle) | +y | 0.19 | 28.1 ± 20.0 | +17.9 | 24.2 ± 8.0 | 4.6 ± 2.4 | +0.6 | 4.4 ± 2.3', '28.1', '{:.1f}', LSa('A_single_base_traverse', 2)),
 ('4.4', 'Table 4: 3 (top) | +z | 0.11 | 33.4 ± 19.8 | +28.6 | 34.0 ± 9.5 | 2.1 ± 1.1 | 0.0 | 2.0 ± 1.1', '33.4', '{:.1f}', LSa('A_single_base_traverse', 3)),
 ('4.4', 'Table 4: 1 (bottom) | -z | 0.58 | 3.7 ± 2.0 | -0.1 | 2.6 ± 1.4 | 2.1 ± 1.2 | 0.0 | 2.0 ± 1.1', '2.6', '{:.1f}', L('A_single_base_traverse', 1)),
 ('4.4', 'Table 4: 2 (middle) | +y | 0.19 | 28.1 ± 20.0 | +17.9 | 24.2 ± 8.0 | 4.6 ± 2.4 | +0.6 | 4.4 ± 2.3', '24.2', '{:.1f}', L('A_single_base_traverse', 2)),
 ('4.4', 'Table 4: 3 (top) | +z | 0.11 | 33.4 ± 19.8 | +28.6 | 34.0 ± 9.5 | 2.1 ± 1.1 | 0.0 | 2.0 ± 1.1', '34.0', '{:.1f}', L('A_single_base_traverse', 3)),
 ('4.4', 'Inverting the two traverses together recovers the lower, middle, and upper flows to 2.1 ± 1.2°, 4.6 ± 2.4°, and 2.1 ± 1.1°, respectively, with intensity biases …', '2.1', '{:.1f}', LSa('C_vertical_pair', 1)),
 ('4.4', '…together recovers the lower, middle, and upper flows to 2.1 ± 1.2°, 4.6 ± 2.4°, and 2.1 ± 1.1°, respectively, with intensity biases of ≤0.6%; …', '4.6', '{:.1f}', LSa('C_vertical_pair', 2)),
 ('4.4', '…the lower, middle, and upper flows to 2.1 ± 1.2°, 4.6 ± 2.4°, and 2.1 ± 1.1°, respectively, with intensity biases of ≤0.6%; damping changes these by …', '2.1', '{:.1f}', LSa('C_vertical_pair', 3)),
 ('4.4', 'Table 4: 2 (middle) | +y | 0.19 | 28.1 ± 20.0 | +17.9 | 24.2 ± 8.0 | 4.6 ± 2.4 | +0.6 | 4.4 ± 2.3', '4.4', '{:.1f}', L('C_vertical_pair', 2)),
 ('4.4', 'On its own, a traverse at z = 12.5 m recovers the highest flow to 3.4° and the lowest to 23°, the reverse of the ground-level result.', '3.4', '{:.1f}', LSa('B_single_elevated_traverse', 3)),
 ('4.4', '…its own, a traverse at z = 12.5 m recovers the highest flow to 3.4° and the lowest to 23°, the reverse of the ground-level result.', '23', '{:.0f}', LSa('B_single_elevated_traverse', 1)),
 ('4.4', '…number of G_K (the ratio of its largest to smallest singular value) drops from 239 to 21.', '239', '{:.0f}', tw['A_single_base_traverse']['operator_diagnostics']['condition_number']),
 ('4.4', '…of G_K (the ratio of its largest to smallest singular value) drops from 239 to 21.', '21', '{:.0f}', tw['C_vertical_pair']['operator_diagnostics']['condition_number']),
 ('4.4', '…near 1 means two flows produce nearly the same field along the traverse) drops from 0.997 for the ground traverse alone to 0.87 for the two traverses together, and the condition number …', '0.997', '{:.3f}', tw['A_single_base_traverse']['operator_diagnostics']['interlayer_correlation_max']),
 ('4.4', '…the same field along the traverse) drops from 0.997 for the ground traverse alone to 0.87 for the two traverses together, and the condition number of G_K (the ratio of its largest …', '0.87', '{:.2f}', tw['C_vertical_pair']['operator_diagnostics']['interlayer_correlation_max']),
 ('4.4', '…∼30°, while the lowest flow magnetized along +y was recovered to 4.7°.', '4.7', '{:.1f}', a3['cases']['T3_swap_L1_L2']['layer_1']['dir_err_mean']),
 ('4.4', '…is 1.0 m from the sensor at its nearest point, and the middle and highest flows are 4.6 m and 9.6 m away, respectively.', '4.6', '{:.1f}', a4['three_layer_nearest_face_m'][1]),
 ('4.4', '…from the sensor at its nearest point, and the middle and highest flows are 4.6 m and 9.6 m away, respectively.', '9.6', '{:.1f}', a4['three_layer_nearest_face_m'][2]),
 ('4.4', 'All three flows are recovered to within 5° (mean) for 6.4 ≲ z₂ ≲ 13.2 m (interpolated) and to within 10° for z₂ ≳ 3.5 m.', '6.4', '{:.1f}', hs['headline']['thresholds']['closed_form_ls']['all_three_flows_within_5deg']['interpolated_min_z2_m']['x']),
 ('4.4', '…for 6.4 ≲ z₂ ≲ 13.2 m (interpolated) and to within 10° for z₂ ≳ 3.5 m.', '3.5', '{:.1f}', hs['headline']['thresholds']['closed_form_ls']['all_three_flows_within_10deg']['interpolated_min_z2_m']['x']),
 ('4.4', 'The worst-recovered flow does best at z₂ = 10 m (3.2°), above which the second traverse moves away from the middle flow and the worst-flow error …', '3.2', '{:.1f}', maxerr['10']),
 ('4.4', '…the second traverse moves away from the middle flow and the worst-flow error grows to 4.5° at 12.5 m and 6.2° at 15 m (500 inversions per height; Table …', '4.5', '{:.1f}', maxerr['12.5']),
 ('4.4', '…from the middle flow and the worst-flow error grows to 4.5° at 12.5 m and 6.2° at 15 m (500 inversions per height; Table 4 gives 4.6° …', '6.2', '{:.1f}', maxerr['15']),
 ('4.4', '…line twice (z₂ = 0.5 m) doubles the data but leaves the middle and upper flows at 21° and 26°.', '21', '{:.0f}', H[0.5]['per_layer_closed_form_ls']['layer_2']['dir_err_deg']['mean']),
 ('4.4', '…= 0.5 m) doubles the data but leaves the middle and upper flows at 21° and 26°.', '26', '{:.0f}', H[0.5]['per_layer_closed_form_ls']['layer_3']['dir_err_deg']['mean']),
 ('4.4', 'A 200 m wall (x = ± 100 m) gives nearly the same result as the 100 m wall (1.9°, 4.4°, and 2.0° with both traverses).', '1.9', '{:.1f}', g7('W100','ground+12.5m',1)),
 ('4.4', '…100 m) gives nearly the same result as the 100 m wall (1.9°, 4.4°, and 2.0° with both traverses).', '2.0', '{:.1f}', g7('W100','ground+12.5m',3)),
 ('4.4', '…(condition number of G_K >10^6) and all three flows are unrecoverable (errors of ∼90°), even with both traverses.', '90', '{:.0f}', min(g7('W250','ground+12.5m',i) for i in (1,2,3))),
 ('4.4', '…is enough; e.g., a 1-km-long wall that ends at x = 50 m, inside the traverse, gives 2.8°, 6.1°, and 2.9° with both traverses, and a wall with a 90° …', '2.8', '{:.1f}', g7('END','ground+12.5m',1)),
 ('4.4', '…is enough; e.g., a 1-km-long wall that ends at x = 50 m, inside the traverse, gives 2.8°, 6.1°, and 2.9° with both traverses, and a wall with a 90° …', '6.1', '{:.1f}', g7('END','ground+12.5m',2)),
 ('4.4', '…with a 90° bend at x = 0, surveyed with 120 m of traverse along each limb, gives 1.4°, 5.0°, and 2.1°.', '1.4', '{:.1f}', g7('BEND','ground+12.5m',1)),
 ('4.4', '…at x = 0, surveyed with 120 m of traverse along each limb, gives 1.4°, 5.0°, and 2.1°.', '5.0', '{:.1f}', g7('BEND','ground+12.5m',2)),
 ('4.4', 'Extending the flows 1 km back into the wall changes these by ≤0.3°.', '0.3', '{:.1f}', max(max(abs(g7(a,'ground+12.5m',i)-g7(b,'ground+12.5m',i)) for i in (1,2,3)) for a,b in (('END','END_deep'),('BEND','BEND_deep')))),
 ('4.4', 'For the 2-km wall, adding a 100 m cross-strike profile to both traverses gives direction errors of 1.1°, 3.7°, and 1.5°, but only because our test magnetizations have no …', '1.1', '{:.1f}', a7['W1000_XPROF']['ground+12.5m']['cross_strike_only']['layer_1']['dir_err_mean']),
 ('4.4', 'For the 2-km wall, adding a 100 m cross-strike profile to both traverses gives direction errors of 1.1°, 3.7°, and 1.5°, but only because our test magnetizations have no …', '3.7', '{:.1f}', a7['W1000_XPROF']['ground+12.5m']['cross_strike_only']['layer_2']['dir_err_mean']),
 ('4.4', 'All three flows are recovered to within 5° (mean) for 6.4 ≲ z₂ ≲ 13.2 m (interpolated) and to within 10° for z₂ ≳ 3.5 m.', '13.2', '{:.1f}', z_hi),
 ('4.4', '…traverse has to span the full exposure; limited to the wall face (|x| ≤ 50 m) it still gives 2.3°, 5.8°, and 2.8°, but limited to |x| ≤ 25 m it gives …', '2.3', '{:.1f}', g7('W50_ELEV_50','ground+12.5m',1)),
 ('4.4', '…to span the full exposure; limited to the wall face (|x| ≤ 50 m) it still gives 2.3°, 5.8°, and 2.8°, but limited to |x| ≤ 25 m it gives 3.3°, …', '5.8', '{:.1f}', g7('W50_ELEV_50','ground+12.5m',2)),
 ('4.4', '…full exposure; limited to the wall face (|x| ≤ 50 m) it still gives 2.3°, 5.8°, and 2.8°, but limited to |x| ≤ 25 m it gives 3.3°, 14.3°, …', '2.8', '{:.1f}', g7('W50_ELEV_50','ground+12.5m',3)),
 ('4.4', '…gives 2.3°, 5.8°, and 2.8°, but limited to |x| ≤ 25 m it gives 3.3°, 14.3°, and 13.9°.', '3.3', '{:.1f}', g7('W50_ELEV_25','ground+12.5m',1)),
 ('4.4', '…5.8°, and 2.8°, but limited to |x| ≤ 25 m it gives 3.3°, 14.3°, and 13.9°.', '14.3', '{:.1f}', g7('W50_ELEV_25','ground+12.5m',2)),
 ('4.4', '…and 2.8°, but limited to |x| ≤ 25 m it gives 3.3°, 14.3°, and 13.9°.', '13.9', '{:.1f}', g7('W50_ELEV_25','ground+12.5m',3)),
 ('5', '…tolerance halved, the combined error is 3.4–5.7° for centered loops and 6.3–9.8° for loops 0.5 m off center (Section 5.3), with slow drift, the unknown …', '9.8', '{:.1f}', rng9(['V_budget_half_offcentre'], f=max)),
 ('5', 'With every tolerance halved, the combined error is 3.4–5.7° for centered loops and 6.3–9.8° for loops 0.5 m off center …', '3.4', '{:.1f}', rng9(['X_budget_half_centred'])),
 ('5', 'With every tolerance halved, the combined error is 3.4–5.7° for centered loops and 6.3–9.8° for loops 0.5 m off center (Section …', '5.7', '{:.1f}', rng9(['X_budget_half_centred'], f=max)),
 ('5', '…by tens of percent, and the combined runs at full, half, and quarter tolerance are biased high by 5–26%.', '5', '{:.0f}', rng9(['U_budget_full_offcentre','V_budget_half_offcentre','X_budget_half_centred','Y_budget_quarter_offcentre'], m='dM_signed_pct')),
 ('5', '…percent, and the combined runs at full, half, and quarter tolerance are biased high by 5–26%.', '26', '{:.0f}', rng9(['U_budget_full_offcentre','V_budget_half_offcentre','X_budget_half_centred','Y_budget_quarter_offcentre'], m='dM_signed_pct', f=max)),
 ('5.1', 'At a fixed ASD, sampling every 1, 2, 5, 10, or 25 cm gives the same mean direction error (3.2–3.3°; Table 3), because each coarser sample averages more of the …', '3.2', '{:.1f}', min(r['dir_err_deg']['mean'] for r in E['drive_around'])),
 ('5.1', '…a fixed ASD, sampling every 1, 2, 5, 10, or 25 cm gives the same mean direction error (3.2–3.3°; Table 3), because each coarser sample averages more of the …', '3.3', '{:.1f}', max(r['dir_err_deg']['mean'] for r in E['drive_around'])),
 ('5.1', 'Our 0.71 nT Hz⁻¹/2 gives ∼3°, and because the errors scale with the ratio of noise to …', '0.71', '{:.2f}', a6['F_estimator_band']['drive_around']['white_asd_nT_per_rtHz_at_1cm_s']),
 ('5.1', "…of the estimator's sensitivity to stationary noise lies at along-track wavelengths of ≳12 m, which at 1 cm s⁻¹ corresponds to periods of ≳20 min, comparable to the …", '12', '{:.0f}', 1/a6['F_estimator_band']['drive_around']['quantiles_cyc_per_m']['k95']),
 ('5.1', 'The drive-around error is 4.5° at L = 1 cm and 10.6° at 5 cm (predicted 10.1°), and it reaches …', '4.5', '{:.1f}', A3['drive_around']['0.01']['none']['dir_err_deg']['mean']),
 ('5.1', 'The drive-around error is 4.5° at L = 1 cm and 10.6° at 5 cm (predicted 10.1°), and it reaches 15–67° for L between 10 cm …', '10.6', '{:.1f}', A3['drive_around']['0.05']['none']['dir_err_deg']['mean']),
 ('5.1', 'The drive-around error is 4.5° at L = 1 cm and 10.6° at 5 cm (predicted 10.1°), and it reaches 15–67° for L between 10 cm and 100 m.', '10.1', '{:.1f}', 3.2*a6['F_estimator_band']['drive_around']['ou_prediction'][3]['asd_ratio']),
 ('5.1', '…at L = 1 cm and 10.6° at 5 cm (predicted 10.1°), and it reaches 15–67° for L between 10 cm and 100 m.', '15', '{:.0f}', min(A3['drive_around'][k]['none']['dir_err_deg']['mean'] for k in ('0.1','0.2','0.5','1.0','2.0','5.0','10.0','20.0','50.0','100.0'))),
 ('5.1', '…at L = 1 cm and 10.6° at 5 cm (predicted 10.1°), and it reaches 15–67° for L between 10 cm and 100 m.', '67', '{:.0f}', max(A3['drive_around'][k]['none']['dir_err_deg']['mean'] for k in ('0.1','0.2','0.5','1.0','2.0','5.0','10.0','20.0','50.0','100.0'))),
 ('5.1', '…offset and a 2 nT local-frame offset in the data, the inversion without offsets fails (mean direction error 91°), and fitting the offsets restores 3.5° and 12%.', '91', '{:.0f}', A2('drive_around','none')),
 ('5.1', '…without offsets fails (mean direction error 91°), and fitting the offsets restores 3.5° and 12%.', '3.5', '{:.1f}', A2('drive_around','rover_world_dc')),
 ('5.1', '…(mean direction error 91°), and fitting the offsets restores 3.5° and 12%.', '12', '{:.0f}', A2('drive_around','rover_world_dc','dM_abs_pct','median')),
 ('5.1', 'For an offset that is strictly constant over the survey, fitting gives 0.4°, but a slow drift is not constant, and with L = 50 and 100 m the fitted …', '0.4', '{:.1f}', A3['drive_around']['inf']['rover_world_dc']['dir_err_deg']['mean']),
 ('5.1', '…a slow drift is not constant, and with L = 50 and 100 m the fitted inversion still gives 20° and 15°, respectively.', '20', '{:.0f}', A3['drive_around']['50.0']['rover_world_dc']['dir_err_deg']['mean']),
 ('5.1', '…and with L = 50 and 100 m the fitted inversion still gives 20° and 15°, respectively.', '15', '{:.0f}', A3['drive_around']['100.0']['rover_world_dc']['dir_err_deg']['mean']),
 ('5.1', '…RMS also stays at 0.99–1.00 for L up to 20 cm, where the mean direction error is 21°, which implies that the fit itself does not reveal this noise.', '21', '{:.0f}', cda['20cm']['dir_err_deg']['mean']),
 ('5.1', "…when the drift's variation about its survey mean is 0.03 nT rms per axis (4.7° at 0.029 nT and 6.8° at 0.058 nT).", '0.029', '{:.3f}', a6['G_gradient_and_drift']['drive_around']['drift_L100m'][3]['rms_about_mean_nT']),
 ('5.1', "…when the drift's variation about its survey mean is 0.03 nT rms per axis (4.7° at 0.029 nT and 6.8° at 0.058 nT).", '4.7', '{:.1f}', a6['G_gradient_and_drift']['drive_around']['drift_L100m'][3]['offsets_fitted']['dir_err_deg']['mean']),
 ('5.1', '…its survey mean is 0.03 nT rms per axis (4.7° at 0.029 nT and 6.8° at 0.058 nT).', '0.058', '{:.3f}', a6['G_gradient_and_drift']['drive_around']['drift_L100m'][4]['rms_about_mean_nT']),
 ('5.1', '…its survey mean is 0.03 nT rms per axis (4.7° at 0.029 nT and 6.8° at 0.058 nT).', '6.8', '{:.1f}', a6['G_gradient_and_drift']['drive_around']['drift_L100m'][4]['offsets_fitted']['dir_err_deg']['mean']),
 ('5.1', 'Linear or cubic drift terms help little for L between 1 and 100 m (errors of 11–61°), and higher-order drift terms absorb the outcrop signal itself.', '11', '{:.0f}', min(A3['drive_around'][k][m]['dir_err_deg']['mean'] for k in ('1.0','2.0','5.0','10.0','20.0','50.0','100.0') for m in ('rover_world_dc_drift','rover_dc_world_poly3'))),
 ('5.1', 'Linear or cubic drift terms help little for L between 1 and 100 m (errors of 11–61°), and higher-order drift terms absorb the outcrop signal itself.', '61', '{:.0f}', max(A3['drive_around'][k][m]['dir_err_deg']['mean'] for k in ('1.0','2.0','5.0','10.0','20.0','50.0','100.0') for m in ('rover_world_dc_drift','rover_dc_world_poly3'))),
 ('5.1', '…mean is ≲0.03 nT rms per axis for a vertical magnetization, compared with the 0.014 nT contributed by white noise at 0.7 nT Hz⁻¹/2 at periods between 20 min and the 40 min …', '0.014', '{:.3f}', 0.707*math.sqrt(1/1200-1/2400)),
 ('5.1', 'With the loops 0.5 m off center, the same drifts do less harm (0.03 nT gives 4.0–7.9° instead of 4.6–11.8°; Table 5).', '7.9', '{:.1f}', rng9(['Z_single_drift'], f=max)),
 ('5.1', 'With both loops centered, a drift of 0.015, 0.02, and 0.03 nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest …', '3.7', '{:.1f}', rng9(['D_drift_0.015nT_dx0'])),
 ('5.1', '…loops centered, a drift of 0.015, 0.02, and 0.03 nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest values …', '6.1', '{:.1f}', rng9(['D_drift_0.015nT_dx0'], f=max)),
 ('5.1', '…of 0.015, 0.02, and 0.03 nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest values for a horizontal …', '4.2', '{:.1f}', rng9(['D_drift_0.02nT_dx0'])),
 ('5.1', '…of 0.015, 0.02, and 0.03 nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest values for a horizontal …', '8.4', '{:.1f}', rng9(['D_drift_0.02nT_dx0'], f=max)),
 ('5.1', '…0.03 nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest values for a horizontal magnetization, and fitting …', '4.6', '{:.1f}', rng9(['D_drift_0.03nT_dx0'])),
 ('5.1', '…nT rms about the survey mean gives 3.7–6.1°, 4.2–8.4°, and 4.6–11.8°, respectively, with the largest values for a horizontal magnetization, and fitting a …', '11.8', '{:.1f}', rng9(['D_drift_0.03nT_dx0'], f=max)),
 ('5.1', 'With the loops 0.5 m off center, the same drifts do less harm (0.03 nT gives 4.0–7.9° instead of 4.6–11.8°; Table 5).', '4.0', '{:.1f}', rng9(['Z_single_drift'])),
 ('5.1', 'With the loops 0.5 m off center, the same drifts do less harm (0.03 nT gives 4.0–7.9° instead of 4.6–11.8°; Table 5).', '7.9', '{:.1f}', rng9(['Z_single_drift'], f=max)),
 ('5.1', '…a horizontal magnetization, and fitting a linear drift over the survey does not help (7.8° for a horizontal magnetization at 0.015 nT).', '7.8', '{:.1f}', B9('H2_drift_0.015nT_dx0_linfit', '+x')),
 ('5.1', 'Even 0.01 nT gives 3.5–6.0° over six magnetization directions with the loops centered …', '3.5', '{:.1f}', rngk('D_drift_0.01nT_dx0')),
 ('5.1', 'Even 0.01 nT gives 3.5–6.0° over six magnetization directions with the loops centered (3.1–4.4° with …', '6.0', '{:.1f}', rngk('D_drift_0.01nT_dx0', f=max)),
 ('5.1', '…gives 3.5–6.0° over six magnetization directions with the loops centered (3.1–4.4° with them 0.5 m off center).', '3.1', '{:.1f}', rngk('D_drift_0.01nT_dx0.5')),
 ('5.1', '…gives 3.5–6.0° over six magnetization directions with the loops centered (3.1–4.4° with them 0.5 m off center).', '4.4', '{:.1f}', rngk('D_drift_0.01nT_dx0.5', f=max)),
 ('5.1', '…once in each direction, doubles the survey time but reduces the error with 0.01 nT of drift to 2.6–4.1° for centered loops.', '2.6', '{:.1f}', rngk('RV_drift_0.01nT_dx0')),
 ('5.1', '…each direction, doubles the survey time but reduces the error with 0.01 nT of drift to 2.6–4.1° for centered loops.', '4.1', '{:.1f}', rngk('RV_drift_0.01nT_dx0', f=max)),
 ('5.2', '…with a second loop at ∼3 m standoff so that the site field can be fitted (70 min more at 1 cm s⁻¹).', '70', '{:.0f}', round((I['loop_1m+loop_3m']['n_stations']-I['loop_1m']['n_stations'])/60, -1)),
 ('5.2', '…tilt error is odd about the loop center, like a gradient, so tilts up to 0.3° leave the direction error at 3.2–4.5°.', '3.2', '{:.1f}', min(J9[(300.0, t)][d]['dir_err_deg']['mean'] for t in (0.003,0.01,0.03,0.1,0.3) for d in DN)),
 ('5.2', '…loop center, like a gradient, so tilts up to 0.3° leave the direction error at 3.2–4.5°.', '4.5', '{:.1f}', max(J9[(300.0, t)][d]['dir_err_deg']['mean'] for t in (0.003,0.01,0.03,0.1,0.3) for d in DN)),
 ('5.2', '…or 1.0 m sensor height, or with the loops 0.5 m off center, a tilt of 0.03° (0.16 nT) gives 3.0–8.0° and 0.1° gives 4.5–21°.', '3.0', '{:.1f}', min(T9[c][0.03][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', '…height, or with the loops 0.5 m off center, a tilt of 0.03° (0.16 nT) gives 3.0–8.0° and 0.1° gives 4.5–21°.', '8.0', '{:.1f}', max(T9[c][0.03][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', '…at 0.25 or 1.0 m sensor height, or with the loops 0.5 m off center, a tilt of 0.03° (0.16 nT) gives 3.0–8.0° and 0.1° gives 4.5–21°.', '0.16', '{:.2f}', 300*math.radians(0.03)),
 ('5.2', '…0.5 m off center, a tilt of 0.03° (0.16 nT) gives 3.0–8.0° and 0.1° gives 4.5–21°.', '4.5', '{:.1f}', min(T9[c][0.1][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', '…a tilt of 0.03° (0.16 nT) gives 3.0–8.0° and 0.1° gives 4.5–21°.', '21', '{:.0f}', max(T9[c][0.1][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', '…three site-field offsets in the three-layer inversions changes the two-traverse errors by ≤1.1° (the middle flow goes from 5.0° to 6.1° at the bend).', '1.1', '{:.1f}', max(abs(a7o(c,i) - g7(c,'ground+12.5m',i)) for c in ('W50','END','BEND','W50_ELEV_50') for i in (1,2,3))),
 ('5.2', '…inversions changes the two-traverse errors by ≤1.1° (the middle flow goes from 5.0° to 6.1° at the bend).', '6.1', '{:.1f}', a7o('BEND',2)),
 ('5.2', '…and a uniform gradient fitted to both traverses, a site field of 30 nT from a dipole 10 km away leaves 2.2°, 4.6°, and 2.1°, and 100 nT from 10 km leaves …', '2.2', '{:.1f}', W9['W50'][(30.0, 10000.0)]['layer_1_dir_err_mean']),
 ('5.2', '…fitted to both traverses, a site field of 30 nT from a dipole 10 km away leaves 2.2°, 4.6°, and 2.1°, and 100 nT from 10 km leaves 2.8°, …', '4.6', '{:.1f}', W9['W50'][(30.0, 10000.0)]['layer_2_dir_err_mean']),
 ('5.2', '…both traverses, a site field of 30 nT from a dipole 10 km away leaves 2.2°, 4.6°, and 2.1°, and 100 nT from 10 km leaves 2.8°, 5.8°, and …', '2.1', '{:.1f}', W9['W50'][(30.0, 10000.0)]['layer_3_dir_err_mean']),
 ('5.2', '…10 km away leaves 2.2°, 4.6°, and 2.1°, and 100 nT from 10 km leaves 2.8°, 5.8°, and 2.6°.', '2.8', '{:.1f}', W9['W50'][(100.0, 10000.0)]['layer_1_dir_err_mean']),
 ('5.2', '…leaves 2.2°, 4.6°, and 2.1°, and 100 nT from 10 km leaves 2.8°, 5.8°, and 2.6°.', '5.8', '{:.1f}', W9['W50'][(100.0, 10000.0)]['layer_2_dir_err_mean']),
 ('5.2', '…4.6°, and 2.1°, and 100 nT from 10 km leaves 2.8°, 5.8°, and 2.6°.', '2.6', '{:.1f}', W9['W50'][(100.0, 10000.0)]['layer_3_dir_err_mean']),
 ('5.2', 'The same fields from 3 km away, which leave 0.03–0.1 nT after the gradient fit, give 11–39° for the middle flow.', '11', '{:.0f}', min(W9['W50'][(B, 3000.0)]['layer_2_dir_err_mean'] for B in (30.0, 100.0))),
 ('5.2', 'The same fields from 3 km away, which leave 0.03–0.1 nT after the gradient fit, give 11–39° for the middle flow.', '39', '{:.0f}', max(W9['W50'][(B, 3000.0)]['layer_2_dir_err_mean'] for B in (30.0, 100.0))),
 ('5.2', 'The same fields from 3 km away, which leave 0.03–0.1 nT after the gradient fit, give 11–39° for the middle flow.', '0.03', '{:.2f}', min(W9['W50'][(B, 3000.0)]['residual_after_gradient_rms_nT'] for B in (30.0, 100.0))),
 ('5.2', 'A tilt of 0.01° (0.05 nT) gives 2.7–5.0°, close to the 2.7–4.6° at 0.003°.', '2.7', '{:.1f}', min(T9[c][0.01][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', 'A tilt of 0.01° (0.05 nT) gives 2.7–5.0°, close to the 2.7–4.6° at 0.003°.', '5.0', '{:.1f}', max(T9[c][0.01][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', 'A tilt of 0.01° (0.05 nT) gives 2.7–5.0°, close to the 2.7–4.6° at 0.003°.', '4.6', '{:.1f}', max(T9[c][0.003][d]['dir_err_deg']['mean'] for c in T9 for d in DN)),
 ('5.2', '…and the loops 0.5 m off center, tilts of 0.03° and 0.1° in 300 nT give 3.1–3.6° and 2.9–3.4°, no worse than no tilt at all.', '3.1', '{:.1f}', rngk('TF_tilt0.03_dx0.5_fit')),
 ('5.2', '…the loops 0.5 m off center, tilts of 0.03° and 0.1° in 300 nT give 3.1–3.6° and 2.9–3.4°, no worse than no tilt at all.', '3.6', '{:.1f}', rngk('TF_tilt0.03_dx0.5_fit', f=max)),
 ('5.2', '…center, tilts of 0.03° and 0.1° in 300 nT give 3.1–3.6° and 2.9–3.4°, no worse than no tilt at all.', '2.9', '{:.1f}', rngk('TF_tilt0.1_dx0.5_fit')),
 ('5.2', '…tilts of 0.03° and 0.1° in 300 nT give 3.1–3.6° and 2.9–3.4°, no worse than no tilt at all.', '3.4', '{:.1f}', rngk('TF_tilt0.1_dx0.5_fit', f=max)),
 ('5.3', '…standoff and may be about as well as a rover can center them on an irregular outcrop, the mean direction error is 11–15°.', '11', '{:.0f}', rng9(['U_budget_full_offcentre'])),
 ('5.3', '…as well as a rover can center them on an irregular outcrop, the mean direction error is 11–15°.', '15', '{:.0f}', rng9(['U_budget_full_offcentre'], f=max)),
 ('5.3', 'Halving every tolerance gives 6.3–9.8°, quartering them gives 4.2–6.0°, and with the …', '6.3', '{:.1f}', rng9(['V_budget_half_offcentre'])),
 ('5.3', 'Halving every tolerance gives 6.3–9.8°, quartering them gives 4.2–6.0°, and with the loops centered and the tolerances halved the error …', '4.2', '{:.1f}', rng9(['Y_budget_quarter_offcentre'])),
 ('5.3', 'Halving every tolerance gives 6.3–9.8°, quartering them gives 4.2–6.0°, and with the loops centered and the tolerances halved the error is 3.4–5.7°.', '6.0', '{:.1f}', rng9(['Y_budget_quarter_offcentre'], f=max)),
 ('5.3', 'However, at half tolerance with the loops off center most of the error (5.5–9.0°) is bias, and because outcrops of one flow that share a …', '5.5', '{:.1f}', rng9(['V_budget_half_offcentre'], k='noise_free_bias')),
 ('5.3', 'However, at half tolerance with the loops off center most of the error (5.5–9.0°) is bias, and because outcrops of one flow that share a magnetization direction would …', '9.0', '{:.1f}', rng9(['V_budget_half_offcentre'], k='noise_free_bias', f=max)),
 ('5.3', 'The noise-free part of the error (1.2–15° in these runs) arises from burial, pose, the site field, the tilt, and the error in the …', '1.2', '{:.1f}', rng9(['U_budget_full_offcentre','V_budget_half_offcentre','X_budget_half_centred','Y_budget_quarter_offcentre'], k='noise_free_bias')),
 ('5.3', 'The noise-free part of the error (1.2–15° in these runs) arises from burial, pose, the site field, the tilt, and the error in the …', '15', '{:.0f}', rng9(['U_budget_full_offcentre','V_budget_half_offcentre','X_budget_half_centred','Y_budget_quarter_offcentre'], k='noise_free_bias', f=max)),
 ('5.3', 'The full tolerances are 0.5 nT of white noise; a 100 nT site field from a dipole placed so that 0.02 nT rms is left after a uniform gradient; a pose error within …', '0.02', '{:.2f}', a9['U_budget_full_offcentre']['-z']['site_residual_rms_nT']),
 ('5.3', '…tolerance with the loops off center, the buried extent, the pose error, and the drift gave 9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with …', '9.2', '{:.1f}', rng9(['Z_single_burial'])),
 ('5.3', '…with the loops off center, the buried extent, the pose error, and the drift gave 9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with …', '15.2', '{:.1f}', rng9(['Z_single_burial'], f=max)),
 ('5.3', '…tolerance with the loops off center, the buried extent, the pose error, and the drift gave 9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with …', '5.6', '{:.1f}', rng9(['Z_single_pose'])),
 ('5.3', '…center, the buried extent, the pose error, and the drift gave 9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with 2.7–3.2° for noise …', '6.8', '{:.1f}', rng9(['Z_single_pose'], f=max)),
 ('5.3', '…extent, the pose error, and the drift gave 9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with 2.7–3.2° for noise alone, while the curved …', '4.0', '{:.1f}', rng9(['Z_single_drift'])),
 ('5.3', '…9.2–15.2°, 5.6–6.8°, and 4.0–7.9°, respectively, compared with 2.7–3.2° for noise alone, while the curved site field, the tilt, and the induced …', '2.7', '{:.1f}', rng9(['Z_single_none'])),
 ('5.3', '…5.6–6.8°, and 4.0–7.9°, respectively, compared with 2.7–3.2° for noise alone, while the curved site field, the tilt, and the induced magnetization …', '3.2', '{:.1f}', rng9(['Z_single_none'], f=max)),
 ('5.3', '…for noise alone, while the curved site field, the tilt, and the induced magnetization each stayed within 2.7–4.3°.', '2.7', '{:.1f}', rng9(['Z_single_site','Z_single_tilt','Z_single_induced'])),
 ('5.3', '…while the curved site field, the tilt, and the induced magnetization each stayed within 2.7–4.3°.', '4.3', '{:.1f}', rng9(['Z_single_site','Z_single_tilt','Z_single_induced'], f=max)),
 ('5.3', '…to 2.9–4.1°, but the extra unknowns add noise such that the total error hardly changes (6.7–8.5° against 6.3–9.8° without it).', '6.7', '{:.1f}', rng9(['F2_half_offcentre_root'])),
 ('5.3', '…but the extra unknowns add noise such that the total error hardly changes (6.7–8.5° against 6.3–9.8° without it).', '8.5', '{:.1f}', rng9(['F2_half_offcentre_root'], f=max)),
 ('5.3', 'Table 5: Loops at 1 and 3 m standoff with site-field offsets and gradient fitted; exposed outcrop only in the model unless a root is fitted. Ranges are over the magnetization directions tested (-z, +x, and 45° inclination; the requirement- and mitigated-set rows add +y, 45° inclination toward +y, and 45° upward). Single-source rows use each source at its full tolerance (80 draws per direction); combined rows use 150 draws, and the requirement and mitigated sets 120. Full tolerances: site-field residual 0.02 nT (100 nT at the outcrop), pose ±3° and ±0.15 m, buried extent 0–1 m, drift 0.03 nT rms about the survey mean, tilt 0.05 nT, induced magnetization with χ_ind = 2 × 10⁻³ removed using a susceptibility known to a factor of 2. Half and quarter tolerances scale each of these (the site-field residual, drift and tilt in nT). Requirement set: pose ±1.5° and ±0.075 m, drift 0.01 nT, site residual 0.02 nT, tilt 0.05 nT, susceptibility to a factor of 1.4. Mitigated set: the requirement set with each loop driven twice in opposite directions (double the survey time) and six heading-dependent tilt terms fitted. Bias is the noise-free part of the direction error; δ M_s is the mean signed intensity error. Standard errors of the mean direction errors are 0.1–0.8°. The single-source rows are separate runs from the cent', '0.1', '{:.1f}', min(r['all_sources']['dir_err_deg']['std']/math.sqrt(r['all_sources']['dir_err_deg']['N']) for k,v in a9.items() if k.startswith(('U_','V_','X_','Y_','Z_','F2','REQ')) for dn,r in v.items() if dn!='config')),
 ('5.3', 'Table 5: Loops at 1 and 3 m standoff with site-field offsets and gradient fitted; exposed outcrop only in the model unless a root is fitted. Ranges are over the magnetization directions tested (-z, +x, and 45° inclination; the requirement- and mitigated-set rows add +y, 45° inclination toward +y, and 45° upward). Single-source rows use each source at its full tolerance (80 draws per direction); combined rows use 150 draws, and the requirement and mitigated sets 120. Full tolerances: site-field residual 0.02 nT (100 nT at the outcrop), pose ±3° and ±0.15 m, buried extent 0–1 m, drift 0.03 nT rms about the survey mean, tilt 0.05 nT, induced magnetization with χ_ind = 2 × 10⁻³ removed using a susceptibility known to a factor of 2. Half and quarter tolerances scale each of these (the site-field residual, drift and tilt in nT). Requirement set: pose ±1.5° and ±0.075 m, drift 0.01 nT, site residual 0.02 nT, tilt 0.05 nT, susceptibility to a factor of 1.4. Mitigated set: the requirement set with each loop driven twice in opposite directions (double the survey time) and six heading-dependent tilt terms fitted. Bias is the noise-free part of the direction error; δ M_s is the mean signed intensity error. Standard errors of the mean direction errors are 0.1–0.8°. The single-source rows are separate runs from the centerin', '0.8', '{:.1f}', max(r['all_sources']['dir_err_deg']['std']/math.sqrt(r['all_sources']['dir_err_deg']['N']) for k,v in a9.items() if k.startswith(('U_','V_','X_','Y_','Z_','F2','REQ')) for dn,r in v.items() if dn!='config')),
 ('5.3', 'With the loops 0.1 m off center, the error is 3.7–5.6° for a buried extent up to 0.5 m and 3.7–7.1° for one up to 1 …', '3.7', '{:.1f}', rngk('REQ_a_dx0.1_b0.5')),
 ('5.3', 'With the loops 0.1 m off center, the error is 3.7–5.6° for a buried extent up to 0.5 m and 3.7–7.1° for one up to 1 m, and with …', '5.6', '{:.1f}', rngk('REQ_a_dx0.1_b0.5', f=max)),
 ('5.3', '…0.1 m off center, the error is 3.7–5.6° for a buried extent up to 0.5 m and 3.7–7.1° for one up to 1 m, and with the loops 0.5 m off center and the buried extent …', '3.7', '{:.1f}', rngk('REQ_b_dx0.1_b1')),
 ('5.3', '…m off center, the error is 3.7–5.6° for a buried extent up to 0.5 m and 3.7–7.1° for one up to 1 m, and with the loops 0.5 m off center and the buried extent known, …', '7.1', '{:.1f}', rngk('REQ_b_dx0.1_b1', f=max)),
 ('5.3', '…3.7–7.1° for one up to 1 m, and with the loops 0.5 m off center and the buried extent known, it is 4.2–5.8°.', '4.2', '{:.1f}', rngk('REQ_c_dx0.5_b0')),
 ('5.3', '…for one up to 1 m, and with the loops 0.5 m off center and the buried extent known, it is 4.2–5.8°.', '5.8', '{:.1f}', rngk('REQ_c_dx0.5_b0', f=max)),
 ('5.3', '…m off center, the error is 3.7–5.6° for a buried extent up to 0.5 m and 3.7–7.1° for one up to 1 m, and with the loops 0.5 m off center and the buried extent known, …', '3.7', '{:.1f}', min(rngk(k) for k in ('REQ_a_dx0.1_b0.5','REQ_b_dx0.1_b1','REQ_c_dx0.5_b0'))),
 ('5.3', '…and 5.2; ∼210 min of driving), the error falls to 3.0–4.9° for a buried extent up to 0.5 m and to 3.0–4.8° with the loops …', '3.0', '{:.1f}', rngk('REQF_a_dx0.1_b0.5')),
 ('5.3', '…and 5.2; ∼210 min of driving), the error falls to 3.0–4.9° for a buried extent up to 0.5 m and to 3.0–4.8° with the loops 0.5 m off …', '4.9', '{:.1f}', rngk('REQF_a_dx0.1_b0.5', f=max)),
 ('5.3', '…driving), the error falls to 3.0–4.9° for a buried extent up to 0.5 m and to 3.0–4.8° with the loops 0.5 m off center and the buried extent known.', '3.0', '{:.1f}', rngk('REQF_c_dx0.5_b0')),
 ('5.3', '…the error falls to 3.0–4.9° for a buried extent up to 0.5 m and to 3.0–4.8° with the loops 0.5 m off center and the buried extent known.', '4.8', '{:.1f}', rngk('REQF_c_dx0.5_b0', f=max)),
 ('5.3', 'A buried extent up to 1 m still gives 3.2–6.9°, with the worst case almost all bias (5.9° of 6.9°), …', '3.2', '{:.1f}', rngk('REQF_b_dx0.1_b1')),
 ('5.3', 'A buried extent up to 1 m still gives 3.2–6.9°, with the worst case almost all bias (5.9° of 6.9°), which indicates …', '6.9', '{:.1f}', rngk('REQF_b_dx0.1_b1', f=max)),
 ('5.3', '…extent up to 1 m still gives 3.2–6.9°, with the worst case almost all bias (5.9° of 6.9°), which indicates that once drift and tilt are handled in this way the …', '5.9', '{:.1f}', rngb('REQF_b_dx0.1_b1', f=max)),
 ('5.3', 'Table 5: Mitigated set, b ≤ 0.5 m | 0.1 | 6 | 3.0–4.9 | 5–10 | 1.9–3.6 | 14 to 19', '3.6', '{:.1f}', rngb('REQF_a_dx0.1_b0.5', f=max)),
 ('5.3', 'Table 5: Mitigated set, burial known | 0.5 | 6 | 3.0–4.8 | 6–10 | 2.0–3.6 | -1 to 0', '3.6', '{:.1f}', rngb('REQF_c_dx0.5_b0', f=max)),
 ('5.3', '…and with the susceptibility still known to a factor of 1.4, the same mitigated set gives 3.8–5.1° for a buried extent up to 0.5 m and 4.1–5.5° with the buried extent …', '3.8', '{:.1f}', rngk('REQF_chi5_a_dx0.1_b0.5')),
 ('5.3', '…with the susceptibility still known to a factor of 1.4, the same mitigated set gives 3.8–5.1° for a buried extent up to 0.5 m and 4.1–5.5° with the buried extent known.', '5.1', '{:.1f}', rngk('REQF_chi5_a_dx0.1_b0.5', f=max)),
 ('5.3', '…1.4, the same mitigated set gives 3.8–5.1° for a buried extent up to 0.5 m and 4.1–5.5° with the buried extent known.', '4.1', '{:.1f}', rngk('REQF_chi5_c_dx0.5_b0')),
 ('5.3', '…the same mitigated set gives 3.8–5.1° for a buried extent up to 0.5 m and 4.1–5.5° with the buried extent known.', '5.5', '{:.1f}', rngk('REQF_chi5_c_dx0.5_b0', f=max)),
 ('5.5', '…passed 5° once the rms field of the background along the traverse exceeded 0.28 times that of the outcrop, and 10° at 0.77 times.', '0.28', '{:.2f}', pooled['bg_to_signal_at_5deg']['x']),
 ('5.5', '…the traverse exceeded 0.28 times that of the outcrop, and 10° at 0.77 times.', '0.77', '{:.2f}', pooled['bg_to_signal_at_10deg']['x']),
 ('5.5', 'Across 82 pooled combinations of block size, intensity range, and direction scatter, the direction error passed …', '82', '{}', pooled['n_points']),
 ('5.5', '…to 0.3 nT m⁻¹ leaves the direction error of a vertical magnetization unchanged (3.5–3.7° with offsets fitted).', '3.5', '{:.1f}', min(r['offsets_fitted']['dir_err_deg']['mean'] for r in a6['G_gradient_and_drift']['drive_around']['gradient'])),
 ('5.5', '…to 0.3 nT m⁻¹ leaves the direction error of a vertical magnetization unchanged (3.5–3.7° with offsets fitted).', '3.7', '{:.1f}', max(r['offsets_fitted']['dir_err_deg']['mean'] for r in a6['G_gradient_and_drift']['drive_around']['gradient'])),
 ('5.5', 'The drive-by has no such symmetry, and its error rises from 8° to 12° at 0.03 nT m⁻¹ and 27° at 0.1 nT m⁻¹, and fitting …', '8', '{:.0f}', a6['G_gradient_and_drift']['drive_by']['gradient'][0]['offsets_fitted']['dir_err_deg']['mean']),
 ('5.5', 'The drive-by has no such symmetry, and its error rises from 8° to 12° at 0.03 nT m⁻¹ and 27° at 0.1 nT m⁻¹, and fitting the three gradient …', '12', '{:.0f}', a6['G_gradient_and_drift']['drive_by']['gradient'][3]['offsets_fitted']['dir_err_deg']['mean']),
 ('5.5', '…and its error rises from 8° to 12° at 0.03 nT m⁻¹ and 27° at 0.1 nT m⁻¹, and fitting the three gradient terms that a straight traverse can resolve …', '27', '{:.0f}', a6['G_gradient_and_drift']['drive_by']['gradient'][4]['offsets_fitted']['dir_err_deg']['mean']),
 ('5.5', '…fitting the three gradient terms that a straight traverse can resolve brings it back to 8°.', '8', '{:.0f}', a6['G_gradient_and_drift']['drive_by']['gradient'][4]['offsets_and_gradient_fitted']['dir_err_deg']['mean']),
 ('5.5', '…the error reaches 5° at a cell-to-cell standard deviation of 0.13 M_ref for independent 1 m cells and 0.04 M_ref for 2 m …', '0.13', '{:.2f}', sd_logu(bg['headline']['crossings']['intensity_range_only']['cell_1m_iso']['dynamic_range_at_5deg']['x'])),
 ('5.5', '…standard deviation of 0.13 M_ref for independent 1 m cells and 0.04 M_ref for 2 m blocks.', '0.04', '{:.2f}', sd_logu(bg['headline']['crossings']['intensity_range_only']['block_2m']['dynamic_range_at_5deg']['x'])),
 ('5.5', '…and only the directions scattered, the error reaches 5° at a scatter of 11° for 1 m cells and 4–8° for blocks 1–10 m across.', '11', '{:.0f}', bg['headline']['crossings']['direction_dispersion_only']['cell_1m_iso']['sigma_theta_deg_at_5deg']['x']),
 ('5.5', '…the error reaches 5° at a scatter of 11° for 1 m cells and 4–8° for blocks 1–10 m across.', '4', '{:.0f}', min(bg['headline']['crossings']['direction_dispersion_only'][b]['sigma_theta_deg_at_5deg']['x'] for b in ('block_1m','block_2m','block_5m','block_10m'))),
 ('5.5', '…the error reaches 5° at a scatter of 11° for 1 m cells and 4–8° for blocks 1–10 m across.', '8', '{:.0f}', max(bg['headline']['crossings']['direction_dispersion_only'][b]['sigma_theta_deg_at_5deg']['x'] for b in ('block_1m','block_2m','block_5m','block_10m'))),
 ('5.5', 'Taken one sweep and block size at a time, the crossing falls at a field ratio of 0.23–0.39, but larger blocks reach that ratio at smaller contrasts.', '0.23', '{:.2f}', min(v['bg_to_signal_at_5deg']['x'] for arm in ('direction_dispersion_only','intensity_range_only') for v in bg['headline']['crossings'][arm].values())),
 ('5.5', 'Taken one sweep and block size at a time, the crossing falls at a field ratio of 0.23–0.39, but larger blocks reach that ratio at smaller contrasts.', '0.39', '{:.2f}', max(v['bg_to_signal_at_5deg']['x'] for arm in ('direction_dispersion_only','intensity_range_only') for v in bg['headline']['crossings'][arm].values())),
 ('5.5', 'The mean direction error was 8° for independent 1 m cells, 10–11° for 1 m columns coherent through the layer, and 18–43° for coherent blocks …', '10', '{:.0f}', min(L['raw']['dir_err_deg']['mean'] for L in bg['sweep_1_as_specified']['block_1m']['levels'].values())),
 ('5.5', 'The mean direction error was 8° for independent 1 m cells, 10–11° for 1 m columns coherent through the layer, and 18–43° for coherent blocks …', '11', '{:.0f}', max(L['raw']['dir_err_deg']['mean'] for L in bg['sweep_1_as_specified']['block_1m']['levels'].values())),
 ('5.5', "The layer sat inside a uniform slab magnetized at the layer's mean and extending ±1.2 km, so that only the heterogeneity differs from the uniform case, and no offsets were …", '1.2', '{:.1f}', bg['meta']['background_slab']['lateral_extent_m']/1000),
 ('5.5', "…edge is itself a strong source; a uniform slab truncated ±57 m from the outcrop produces 0.07 nT rms along the drive-around, more than half the outcrop's own field.", '0.07', '{:.2f}', bg['meta']['padding_convergence_study_uniform_field_rms_nT']['+/-57m']),
 ('5.5', 'At the surface, the error reaches 5° at a cell-to-cell standard deviation of 0.14 M_ref for 1 m cells and 0.06 M_ref for 2 m …', '0.14', '{:.2f}', M29['depth_0m:cell_1m']['sd_at_5deg']),
 ('5.5', '…at a cell-to-cell standard deviation of 0.14 M_ref for 1 m cells and 0.06 M_ref for 2 m and 5 m blocks, so the second loop and the fitted terms do not …', '0.06', '{:.2f}', M29['depth_0m:block_2m']['sd_at_5deg']),
 ('5.5', '…at a cell-to-cell standard deviation of 0.14 M_ref for 1 m cells and 0.06 M_ref for 2 m and 5 m blocks, so the second loop and the fitted terms do not …', '0.06', '{:.2f}', M29['depth_0m:block_5m']['sd_at_5deg']),
 ('5.5', 'Under 1 m of cover, 1 m cells never push the error past 5° (≤4.0° at a standard deviation of 0.25 M_ref), and 2 m and 5 m blocks reach …', '4.0', '{:.1f}', max(r['dir_err_deg']['mean'] for r in M29['depth_1m:cell_1m']['rows'])),
 ('5.5', '…at a standard deviation of 0.25 M_ref), and 2 m and 5 m blocks reach 5° at 0.15 and 0.09 M_ref.', '0.15', '{:.2f}', M29['depth_1m:block_2m']['sd_at_5deg']),
 ('5.5', '…deviation of 0.25 M_ref), and 2 m and 5 m blocks reach 5° at 0.15 and 0.09 M_ref.', '0.09', '{:.2f}', M29['depth_1m:block_5m']['sd_at_5deg']),
 ('5.5', 'Under 3 m of cover, no case we tested reaches 5° (≤4.4°).', '4.4', '{:.1f}', max(r['dir_err_deg']['mean'] for k in ('cell_1m','block_2m','block_5m') for r in M29[f'depth_3m:{k}']['rows'])),
 ('5.5', 'A meter of regolith over it relaxes the tolerance for coherent blocks by a factor of 1.6 to 2.8, and 3 m of regolith removes the risk for the cases we tested.', '1.6', '{:.1f}', M29['depth_1m:block_5m']['sd_at_5deg']/M29['depth_0m:block_5m']['sd_at_5deg']),
 ('5.5', 'A meter of regolith over it relaxes the tolerance for coherent blocks by a factor of 1.6 to 2.8, and 3 m of regolith removes the risk for the cases we tested.', '2.8', '{:.1f}', M29['depth_1m:block_2m']['sd_at_5deg']/M29['depth_0m:block_2m']['sd_at_5deg']),
 ('5.5', 'Left uncorrected, it biases the recovered direction by 3.0° and 9.0° (noise-free, two loops, mean over …', '3.0', '{:.1f}', sum(N9[(100.0, 2e-3)][d]['mean'] for d in DN)/3),
 ('5.5', 'Left uncorrected, it biases the recovered direction by 3.0° and 9.0° (noise-free, two loops, mean over directions).', '9.0', '{:.1f}', sum(N9[(300.0, 2e-3)][d]['mean'] for d in DN)/3),
 ('5.5', 'For χ_ind = 2 × 10⁻³, the induced magnetization is 0.07 M_ref in a 100 nT site field and 0.20 M_ref in 300 nT.', '0.07', '{:.2f}', N9[(100.0, 2e-3)]['induced_over_Mref']),
 ('5.5', '…10⁻³, the induced magnetization is 0.07 M_ref in a 100 nT site field and 0.20 M_ref in 300 nT.', '0.20', '{:.2f}', N9[(300.0, 2e-3)]['induced_over_Mref']),
]


# v1.1: numbers of Section 4.3 that changed when the skate-park mesh was centred on the outcrop (review finding
# Y6-04; engine/skatepark_1cm_dr_sweep.build_mesh). The companion paper still prints the v1.0.0 values, so each row
# here records (section, sentence, value printed in the paper, value from the corrected results). count() accepts
# the corrected value for these rows and reports how many there are; count(strict=True) compares with the paper only.
SUPERSEDED_Y6_04 = [
 ('4.3', '…(configuration A), the intensity is biased high, with a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+45', '+47'),
 ('4.3', '…is biased high, with a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+20', '+21'),
 ('4.3', '…a mean signed intensity error of +45% at Δ r = 1 m, +20% at 2 m, and +11% at 3 m.', '+11', '+12'),
 ('4.3', 'For Δ r 5 m, the median intensity error is ≤5.3%, and at 8–10 m it is 3.3–3.7%, comparable to the isolated value of 3.7%.', '5.3', '5.4'),
 ('4.3', 'The residual RMS is 1.002 at Δ r = 1 m and 1.00 elsewhere, within the 0.008 scatter expected for 7197 …', '1.002', '1.001'),
 ('4.3', 'The median direction error is 4.4° at Δ r = 1 m and 2.9–3.2° from 2 to 10 m, which indicates that neighbors …', '4.4', '4.3'),
 ('4.3', 'The median direction error is 4.4° at Δ r = 1 m and 2.9–3.2° from 2 to 10 m, which indicates that neighbors that record the opposite polarity do …', '3.2', '3.1'),
 ('4.3', '…(configuration B), the intensity bias disappears, with a median intensity error of 3.0–4.4% and a mean signed error between -3.1% and +0.2% at every Δ r, because the …', '4.4', '4.5'),
 ('4.3', '…all four neighbors magnetized along +x or +y at Δ r = 1 m, the median direction error is 10.1° and 6.9°, respectively, and with each neighbor in an …', '10.1', '10.5'),
 ('4.3', '…along +x or +y at Δ r = 1 m, the median direction error is 10.1° and 6.9°, respectively, and with each neighbor in an independent random direction …', '6.9', '7.4'),
 ('4.3', '…6.9°, respectively, and with each neighbor in an independent random direction the median is 6.8° and the 95th percentile 13.8°.', '6.8', '7.3'),
 ('4.3', '…each neighbor in an independent random direction the median is 6.8° and the 95th percentile 13.8°.', '13.8', '14.5'),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '3.7', '3.8'),
 ('4.3', 'These fall to 3.7–4.6° at Δ r = 2 m and to 2.8–3.2°, the isolated value, at 5 m.', '4.6', '4.7'),
 ('4.3', '…(means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.3', '4.2'),
 ('4.3', '…fields are partly correlated along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '23', '25'),
 ('4.3', '…fields are partly correlated along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '29', '31'),
 ('4.3', '…along the loop, the median intensity error rises to 23–29% at Δ r = 1 m, 16–18% at 2 m, and 12–13% at 5 m.', '18', '19'),
 ('4.3', '…the central outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.1', '5.3'),
 ('4.3', '…outcrop of 3.1–4.1° in every configuration at Δ r = 1 m (means 5.1–5.8°, just above the goal) and a median signed intensity error within ±5.3%.', '5.8', '6.7'),
 ('4.3', 'At Δ r = 2 m the means are 3.9–4.3°.', '3.9', '4.0'),
]
_SUPERSEDED = {(sec, s, paper): new for sec, s, paper, new in SUPERSEDED_Y6_04}


def numbers(section=None):
    """Print the numbers quoted in one section of the paper (e.g. "4.2"), or in every section."""
    for key in ([section] if section else SECTIONS):
        print(f"\n{SECTIONS[key]}" if key == "Abstract" else f"\nSection {key}. {SECTIONS[key]}")
        for sec, sentence, paper, fmt, value in ROWS:
            if sec == key:
                got = fmt.format(value)
                exp = expected(sec, sentence, paper)
                if got.lstrip("+") != exp.lstrip("+"):
                    flag = f"   <-- the paper says {paper}"
                elif exp != paper:
                    flag = f"   (paper says {paper}; v1.1 skate-park mesh fix)"
                else:
                    flag = ""
                print(f"  {got:>7s}  {sentence}{flag}")


def expected(sec, sentence, paper):
    """The value the stored results should give: the paper's, or the v1.1 corrected one (SUPERSEDED_Y6_04)."""
    return _SUPERSEDED.get((sec, sentence, paper), paper)


def count(strict=False):
    """(numbers that match, numbers checked). Rows superseded by the v1.1 skate-park mesh fix match their corrected
    value; with strict=True every row is compared with the value printed in the companion paper."""
    ok = sum(fmt.format(value).lstrip("+") == (paper if strict else expected(sec, s, paper)).lstrip("+")
             for sec, s, paper, fmt, value in ROWS)
    return ok, len(ROWS)
