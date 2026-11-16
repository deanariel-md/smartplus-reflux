# Code for: Continuously recorded gastroesophageal reflux tracks spontaneous breathing and sedation in ventilated patients

Analysis code for a cohort study of continuous feeding-tube reflux recordings in 45 ventilated ICU patients (smART+ trial). The code turns the device tables and clinical tables into every number, table and figure in the manuscript and its Additional file. It contains no patient data.

## Data

The data are patient-level and are available from the corresponding author on reasonable request. The scripts expect the layout below. Set `DATA_DIR` (data root), `OUT_DIR` (results root), and optionally `T0_DIR` and `SOURCE_DIR` before running; the defaults and the file names of the source files are in `scripts/config.py`.

```
DATA_DIR/
  folder_to_patient.csv            device folder -> study patient number
  t0/RMC1-<id>/*.parquet           full-rate device logs, one Parquet file per patient-hour (T0_DIR)
  t2/minute/<folder>.parquet       one row per patient-minute
  t3/reflux_runs_raw/              contiguous reflux runs, one Parquet file per patient
  t3/reflux_events_7s/             reflux episodes after the drop-and-merge rule, one Parquet file per patient
  t3/balloon_inflations/           balloon inflation runs, one Parquet file per patient
  validation/                      device episode counts against the manufacturer's daily report
  source/                          trial source files (SOURCE_DIR), see below
```

Columns the scripts read:

- `t2/minute`: `folder_id`, `minute_pc` (timestamp), `monitor_s`, `covered_s`, `ree_valid_frac`, `ree_kcal_d`, `vco2_ml_min`, `rsbi`, `peep`, `pip`, `rr_total`, `mv_exp`, `vte`, `etco2`, `cdyn`, `fpf20_running_s`, plus seconds per feeding-tube state, minor and massive onsets, balloon, drain, feed and urine columns.
- `t3` episode tables: `folder_id`, `type` (minor or massive), `start_t`, `end_t`, `in_state_s`, `drained_ml`, `balloon_req_max`. The run table also carries `state` (11 minor, 12 massive, 14 drainage open), `dur_s` and the drain-open fraction.
- `validation/`: `reflux_daily_comparison.csv` (per patient-day report and device counts), `reflux_validation_summary.csv` (agreement by clock and type).

The `t0` tables are produced from the manufacturer's raw logs by a conversion step: it keeps the analysed columns, parses the timestamp format of each file, and joins the seven subsystem files of each hour by row, or by nearest packet count when their row counts differ. That conversion step is not part of this release. The scripts here start from `t0`: `build_t2_t3.py` builds the minute grid and the run tables, and `build_reflux_events.py` applies the episode rule (runs shorter than 1.5 s dropped, then runs of the same type at most 7.5 s apart merged). `verify_rebuild.py` rebuilds both into a temporary folder and compares them with the tables under `DATA_DIR`.

Source files under `SOURCE_DIR` (xlsx unless noted; names are set in `scripts/config.py`):

| Config name | Content |
|---|---|
| `MASTER_WORKBOOK` | trial master workbook, sheet " Patients info" (baseline, daily clinical data, outcomes) |
| `ECRF_MEDICAL_HISTORY` | eCRF medical-history export, sheet "Study results" (gastrointestinal fields only) |
| `ECRF_FEEDING_LOG` | eCRF treated-arm daily feeding log |
| `FIO2_PEEP_DIR` | `Treatment patients.zip`, `Control patients.zip`: hospital FiO2 and PEEP records |
| `DOSE_WORKBOOK` | analysis workbook, sheet "Treated Group" (daily sedative and analgesic doses) |
| `DEVICE_DEFICIENCY_EXPORT` | manufacturer's device-deficiency export, sheet "Data" |
| `ADVERSE_EVENT_TEXT` | text extract of the adverse-event table of the trial report (txt) |

## Software

Python 3.12, DuckDB 1.5.6, pyarrow 25 or later, pandas, NumPy, SciPy, statsmodels, lifelines, scikit-learn, matplotlib, openpyxl. `requirements.txt` pins the versions that were recorded; install with `pip install -r requirements.txt`. Run with `PYTHONUTF8=1` (the run scripts set it). Mixed Poisson models use `scripts/glmm_poisson.py`, a random-intercept model fitted by adaptive Gauss-Hermite quadrature with cluster-robust standard errors.

## Run

```
# PowerShell
$env:DATA_DIR = "<data folder>"; $env:OUT_DIR = "<results folder>"
.\run_all.ps1

# bash
DATA_DIR=/path/to/data OUT_DIR=/path/to/results ./run_all.sh
```

Both scripts run the steps below in order and stop at the first error. Tables are written to `OUT_DIR/descriptive/tables`, `OUT_DIR/explore` and `OUT_DIR/association`, and figures to `OUT_DIR/paper_figures`. File names such as `18_sedation_dose.csv` refer to the `OUT_DIR/explore` folder unless stated.

## Run order, outputs and manuscript items

| # | Script | Main outputs | Manuscript item |
|---|---|---|---|
| 1 | `build_t2_t3.py` | `t2/minute`, `t3/reflux_runs_raw`, `t3/balloon_inflations`, `t3/pump_paused` | Inputs to all device analyses |
| 2 | `build_reflux_events.py` | `t3/reflux_events_7s` | Reflux episode definition (Methods) |
| 3 | `check_balloon_coupling.py` | `validation/balloon_coupling_*.csv` | Fig. S1a |
| 4 | `check_drainage_rule.py` | `validation/drainage_rule_by_era.csv` | Fig. S1a |
| 5 | `build_reflux_axes.py` (primary run, then variants `_raw`, `_art15`, `_art60`, `_noart`, `_noera1`, `_covered`) | `t4/reflux_axes_patient_day`, `reflux_axes_patient`, `artefact_minute_list`, `cutpoints.json`, landmark files | Cohort and signal (Results); sensitivity analyses |
| 6 | `device_fault_days.py` | `validation/device_fault_days.csv`, `14_device_fault_days.csv` | Sensitivity analysis |
| 7 | `build_reflux_axes.py --exclude-days ... --tag _nofault` | `t4/*_nofault` | Sensitivity analysis |
| 8 | `extract_tube_positions.py` | `t3/tube_events.csv`, `t3/tube_position_segments.parquet` | Tube repositioning (Results, part 2) |
| 9 | `build_reflux_drain.py` | `t3/reflux_events_7s_drain.parquet`, `drain_runs`, `pump_off_runs` | Figs 1, 2; Table S3 |
| 10 | `build_arr.py` | `t3/arr_minute.parquet` | Feeding results (part 4) |
| 11 | `describe_reflux.py` | `descriptive/tables/1_coverage.csv` to `9_sensitivity.csv`, `4_course_over_days_glmm.csv`, `models.txt` | Table 1 (monitoring rows); Fig. 1c; device-day trend |
| 12 | `describe_tube_position.py` | `descriptive/tables/13_*.csv` | Tube repositioning |
| 13 | `benchmark_bar_lavie.py` | `descriptive/tables/15_benchmark_bar_lavie.csv` | Verification (Methods, Additional file 1) |
| 14 | `build_clinical_tables.py` | `clinical/baseline_patient.csv`, `patient_day_covariates.parquet`, `day_alignment.csv`, `clinical/outcomes/*` | Inputs to all clinical analyses |
| 15 | `build_energy_sources.py` | `clinical/outcomes/iv_hidden_kcal.parquet`, `pn_days.parquet` | Feeding sensitivity analyses |
| 16 | `describe_cohort.py` | `descriptive/tables/10_baseline.csv` | Table 1 |
| 17 | `build_landmark_exposure.py` | `analysis/landmark_exposure.csv` | Part 3 and 4 exposure (days 1-3) |
| 18 | `build_fio2_peep.py` | `clinical/outcomes/fio2_peep_hourly.parquet`, `vac_events.csv` | Fig. 4b (FiO2 and PEEP rows); Fig. S1c |
| 19 | `association_landmark.py` | `association/spearman.csv`, `supportive_models.csv`, `lagged_mixed_model.csv`, `energy_source_sensitivity.csv`, `dropout_comparison.csv` | Feeding results (part 4) |
| 20 | `describe_outcomes.py` | `association/outcome_summary.csv`, `adequacy_by_bi2.csv` | Feeding results (part 4) |
| 21 | `explore_reflux.py` | `1_drivers_within.csv` to `4_rise.csv`, `frame_patient_day.parquet` | Part 2 (course, feeding); input to later steps |
| 22 | `explore_ventilation.py` | `5_ventilation_*.csv` | Fig. 4a-b; part 3 |
| 23 | `explore_extubation.py` | `6_extubation_*.csv`, `6_breathing_glmm.csv` | Fig. 2d (minute ventilation); part 3 |
| 24 | `explore_infection.py` | `7_infection_event_study.csv`, `7_infection_landmark.csv`, `clinical/outcomes/adverse_events_ctr.csv` | Fig. S2a |
| 25 | `explore_infection_placebo.py` | `7_infection_robustness.csv` | Fig. S2b |
| 26 | `explore_infection_more.py` | `9_*.csv` | Fig. 4b (culture row); Additional file 1 |
| 27 | `explore_oxygenation.py` | `10_*.csv` | Fig. 4b (FiO2 and PEEP rows) |
| 28 | `explore_biomarker_metrics.py` | `12_biomarker_reliability.csv`, `12_snapshot_vs_continuous.csv` | Fig. S3 |
| 29 | `explore_extubation_discrimination.py` | `13_extubation_discrimination.csv` | Part 3 (Harrell's C) |
| 30 | `explore_peri_event.py` | `14_peri_event_curves.csv`, `14_peri_event_windows.csv` | Fig. 2a-c; Table S3 |
| 31 | `explore_extubation_alignment.py` | `15_extubation_alignment.csv` | Part 3 (extubation alignment) |
| 32 | `explore_arr.py` | `17_arr.csv` | Part 4 (active residual release) |
| 33 | `explore_sedation_dose.py` | `18_sedation_dose.csv` | Fig. 2d, Fig. 3, Fig. 4b; Table S1 |
| 34 | `explore_artefact_check.py` | `20_artefact_check.csv` | Table S3 |
| 35 | `check_assumptions.py` | `21_assumption_checks.csv` | Table S2; Table S3 (within-patient rows); Fig. 4b |
| 36 | `paper_extra_numbers.py` | `19_paper_extra.csv` | Part 4 (feeding adequacy) |
| 37 | `fig_paper.py` | `paper_figures/Fig1-4`, `FigS1-S3` (.png, .pdf, .jpg) | Figs 1-4, Figs S1-S3 |

`scripts/config.py` holds all paths; `scripts/glmm_poisson.py` is imported by the analysis scripts.

## Notes

- Several scripts exclude one randomised patient (folder `002`) and a few folders without a balloon; these study-folder numbers are set at the top of the scripts that use them.

## Verification

`scripts/verify_events.py` applies the episode rule (`scripts/build_reflux_events.py`) to the raw reflux runs and compares the result with the episode table used in the paper. It reproduces all 24,530 minor and 8,239 massive episodes exactly.
