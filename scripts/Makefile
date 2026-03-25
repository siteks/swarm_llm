# Makefile for swarm analysis
# Usage: make -f ../scripts/Makefile <target>
# Run from an analysis directory containing swarm_*.json files

SCRIPTS_DIR := ../scripts/
PROMPTS_DIR := ../prompts/

# Find all JSON files (exclude .vocab.json etc.)
JSON_FILES := $(wildcard swarm_*.json)

# Derive storage, analysis, stats, report file names
STORAGE_FILES := $(JSON_FILES:.json=_storage.md)
ANALYSIS_FILES := $(JSON_FILES:.json=_storage_raw_analysis.md)
NARRATIVE_FILES := $(JSON_FILES:.json=_storage_raw_narrative.md)
REPORT_FULL_FILES := $(JSON_FILES:.json=_report_full.md)
STORAGE_RAW_FILES := $(JSON_FILES:.json=_storage_raw.txt)

.PHONY: all storage storage-raw analysis narrative stats report_full clean help

all: summary.csv storage storage-raw analysis narrative report_full all_reports.md

help:
	@echo "Targets:"
	@echo "  summary.csv  - Regenerate summary.csv from all JSON files"
	@echo "  storage      - Create *_storage.md files from JSON files"
	@echo "  storage-raw  - Create *_storage_raw.txt (raw text, no metadata)"
	@echo "  analysis     - Create *_storage_analysis.md files from storage files"
	@echo "  narrative    - Create *_storage_narrative.md files from storage files"
	@echo "  report_full  - Create *_report_full.md combining summary, stats, and analysis, including narrative"
	@echo "  clean        - Remove generated files"

# summary.csv depends on all JSON files
summary.csv: $(JSON_FILES)
	python3 $(SCRIPTS_DIR)summarize_runs.py $(JSON_FILES) -o $@

# Pattern rule: create *_storage.md from *.json
%_storage.md: %.json
	python3 $(SCRIPTS_DIR)extract_memlog.py $<

# Pattern rule: create *_storage_raw.txt from *.json
%_storage_raw.txt: %.json
	python3 $(SCRIPTS_DIR)extract_memlog.py --raw --sep 3 $<

# Pattern rule: create *_storage_analysis.md from *_storage.md
%_storage_raw_analysis.md: %_storage_raw.txt
	python3 $(SCRIPTS_DIR)analyse_markdown.py -m gpro \
		-s $(PROMPTS_DIR)system_expert4.txt \
		-u $(PROMPTS_DIR)user_expert4.txt \
		$<

%_storage_raw_narrative.md: %_storage_raw.txt
	python3 $(SCRIPTS_DIR)analyse_markdown.py -m gpro \
		-s $(PROMPTS_DIR)system_narrativist1.txt \
		-u $(PROMPTS_DIR)user_narrativist1.txt \
		--suffix "_narrative" \
		$<

# Pattern rule: create *_report_full.md from stats, analysis, and narrative
%_report_full.md: %_storage_raw_analysis.md %_storage_raw_narrative.md summary.csv
	python3 $(SCRIPTS_DIR)generate_report.py -n $*.json

# Phony targets to build all storage/analysis/stats/report files
storage: $(STORAGE_FILES)

storage-raw: $(STORAGE_RAW_FILES)

analysis: $(ANALYSIS_FILES)

narrative: $(NARRATIVE_FILES)

report_full: $(REPORT_FULL_FILES)

all_reports.md: $(REPORT_FULL_FILES)
	../scripts/concat.sh $(REPORT_FULL_FILES) > all_reports.md

clean:
	rm -f summary.csv *_storage.md *_storage_raw.txt *_storage_analysis.md *_storage_narrative.md *_report_full.md
