#!/bin/bash
set -e
PY=/Users/satyabratamishra/.local/bin/python3
$PY run_pipeline.py
$PY scripts/build_report.py
$PY scripts/build_slides.py
$PY -m pytest tests/ -q
for n in notebooks/*.ipynb; do jupyter nbconvert --execute "$n" --to notebook --inplace; done
echo ALL DONE
