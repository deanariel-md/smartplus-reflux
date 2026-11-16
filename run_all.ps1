# Runs the full analysis in order. Set DATA_DIR and OUT_DIR first (defaults: .\data and .\results).

# Optional: PYTHON (interpreter, default "python"), T0_DIR, SOURCE_DIR (see scripts\config.py).

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot

Set-Location $Root

if (-not $env:DATA_DIR) { $env:DATA_DIR = Join-Path $Root "data" }

if (-not $env:OUT_DIR) { $env:OUT_DIR = Join-Path $Root "results" }

$env:PYTHONUTF8 = "1"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }



function Run {

    param([string]$Script, [string[]]$ScriptArgs = @())

    Write-Host ""

    Write-Host "=== $Script $($ScriptArgs -join ' ')"

    & $Python "scripts\$Script.py" @ScriptArgs

    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }

}



Run build_t2_t3

Run build_reflux_events

Run check_balloon_coupling

Run check_drainage_rule

Run build_reflux_axes

Run build_reflux_axes @('--events', 'raw', '--tag', '_raw')

Run build_reflux_axes @('--artefact-min', '15', '--tag', '_art15')

Run build_reflux_axes @('--artefact-min', '60', '--tag', '_art60')

Run build_reflux_axes @('--artefact-min', '0', '--tag', '_noart')

Run build_reflux_axes @('--exclude-era', '1', '--tag', '_noera1')

Run build_reflux_axes @('--denominator', 'covered', '--tag', '_covered')

Run device_fault_days

Run build_reflux_axes @('--exclude-days', (Join-Path $env:DATA_DIR "validation\device_fault_days.csv"), '--tag', '_nofault')

Run extract_tube_positions

Run build_reflux_drain

Run build_arr

Run describe_reflux

Run describe_tube_position

Run benchmark_bar_lavie

Run build_clinical_tables

Run build_energy_sources

Run describe_cohort

Run build_landmark_exposure

Run build_fio2_peep

Run association_landmark 

Run describe_outcomes

Run explore_reflux

Run explore_ventilation

Run explore_extubation

Run explore_infection

Run explore_infection_placebo

Run explore_infection_more

Run explore_oxygenation

Run explore_biomarker_metrics

Run explore_extubation_discrimination

Run explore_peri_event

Run explore_extubation_alignment

Run explore_arr

Run explore_sedation_dose

Run explore_artefact_check

Run check_assumptions

Run paper_extra_numbers




Write-Host ""

Write-Host "Done."

