"""Paths used by every script. Set them with environment variables; nothing else is hard-coded.

DATA_DIR    Root of the data tree (device tables, clinical tables, analysis tables).
            Default: <repository>/data
OUT_DIR     Root for results: tables, model output and figures.
            Default: <repository>/results
T0_DIR      Full-rate device logs, one sub-folder per patient (RMC1-<id>/*.parquet).
            Default: DATA_DIR/t0
SOURCE_DIR  Trial source files read once by the build_* scripts (see README, "Source files").
            Default: DATA_DIR/source
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("DATA_DIR", _REPO / "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", _REPO / "results"))
T0_DIR = Path(os.environ.get("T0_DIR", DATA_DIR / "t0"))
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", DATA_DIR / "source"))

# Source files under SOURCE_DIR (file names are fixed here; rename your copies to match).
MASTER_WORKBOOK = SOURCE_DIR / "master_workbook.xlsx"                  # sheet " Patients info"
ECRF_MEDICAL_HISTORY = SOURCE_DIR / "ecrf_medical_history.xlsx"        # sheet "Study results"
ECRF_FEEDING_LOG = SOURCE_DIR / "ecrf_feeding_log.xlsx"                # treated-arm daily feeding log
ECRF_MEDICATION_LOG = SOURCE_DIR / "ecrf_medication_log.xlsx"          # sheet "Medication_Record_Log"
FIO2_PEEP_DIR = SOURCE_DIR / "fio2_peep"                               # "Treatment patients.zip", "Control patients.zip"
DOSE_WORKBOOK = SOURCE_DIR / "dose_workbook.xlsx"                      # sheet "Treated Group"
DEVICE_DEFICIENCY_EXPORT = SOURCE_DIR / "device_deficiencies.xlsx"     # sheet "Data"
ADVERSE_EVENT_TEXT = SOURCE_DIR / "adverse_events_table.txt"           # text extract of the adverse-event table
