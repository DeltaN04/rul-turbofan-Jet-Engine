PY=/Users/satyabratamishra/.local/bin/python3
all: data eda train evaluate report
data:
	$(PY) -c "from src.data import load_split; [load_split('data/raw',f) for f in ['FD001','FD002','FD003','FD004']]"
eda:
	$(PY) run_pipeline.py
train:
	$(PY) run_pipeline.py
evaluate:
	$(PY) -m pytest tests/ -q
report:
	@echo "report built via scripts/build_report.py, slides via scripts/build_slides.py"
