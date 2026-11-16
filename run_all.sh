#!/usr/bin/env bash
# Runs the full analysis in order. Set DATA_DIR and OUT_DIR first (defaults: ./data and ./results).
# Optional: PYTHON (interpreter, default python3), T0_DIR, SOURCE_DIR (see scripts/config.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export OUT_DIR="${OUT_DIR:-$ROOT/results}"
export PYTHONUTF8=1
PYTHON="${PYTHON:-python3}"

run() {
  local script="$1"; shift
  echo
  echo "=== $script $*"
  "$PYTHON" "scripts/$script.py" "$@"
}

run build_t2_t3
run build_reflux_events
run check_balloon_coupling
run check_drainage_rule
run build_reflux_axes
run build_reflux_axes --events raw --tag _raw
run build_reflux_axes --artefact-min 15 --tag _art15
run build_reflux_axes --artefact-min 60 --tag _art60
run build_reflux_axes --artefact-min 0 --tag _noart
run build_reflux_axes --exclude-era 1 --tag _noera1
run build_reflux_axes --denominator covered --tag _covered
run device_fault_days
run build_reflux_axes --exclude-days "$DATA_DIR/validation/device_fault_days.csv" --tag _nofault
run extract_tube_positions
run build_reflux_drain
run build_arr
run describe_reflux
run describe_tube_position
run benchmark_bar_lavie
run build_clinical_tables
run build_energy_sources
run describe_cohort
run build_landmark_exposure
run build_fio2_peep
run association_landmark
run describe_outcomes
run explore_reflux
run explore_ventilation
run explore_extubation
run explore_infection
run explore_infection_placebo
run explore_infection_more
run explore_oxygenation
run explore_biomarker_metrics
run explore_extubation_discrimination
run explore_peri_event
run explore_extubation_alignment
run explore_arr
run explore_sedation_dose
run explore_artefact_check
run check_assumptions
run paper_extra_numbers

echo
echo "Done."
