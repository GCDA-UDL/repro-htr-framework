# make all            regenerate the statistics, tables and figure data in out/
# make check PAPER=.. compare them with the article sources (a checkout of the paper repository)
# The per-line scores on CatK12HW-actions (results/perline/) are available on request (see README). With them,
# out/lines.json is recomputed by the hierarchical bootstrap; without them, the captured output
# results/aggregated/lines.json is used.
PY ?= python
PAPER ?= /paper

all: out/holm_family.json out/lines.json out/expected.json

out/holm_family.json: analysis/holm_family.py results/kraken/*.csv
	$(PY) analysis/holm_family.py

ifneq (,$(wildcard results/perline/lines.csv))
out/lines.json: analysis/lines_analysis.py results/perline/lines.csv results/perline/cells.json
	$(PY) analysis/lines_analysis.py
else
out/lines.json: results/aggregated/lines.json
	mkdir -p out && cp results/aggregated/lines.json out/lines.json
	@echo "results/perline/ not present: using the captured bootstrap output results/aggregated/lines.json"
endif

out/expected.json: analysis/tables.py out/holm_family.json out/lines.json
	$(PY) analysis/tables.py

check: all
	$(PY) analysis/check_manuscript.py $(PAPER)

# with the per-line scores: recompute the bootstrap and check it equals the captured output
check-bootstrap:
	$(PY) analysis/lines_analysis.py && cmp out/lines.json results/aggregated/lines.json && echo "bootstrap identical to results/aggregated/lines.json"

clean:
	rm -rf out/holm_family.* out/lines.* out/expected.json out/tables.md out/prose_values.json out/figures out/check_manuscript.md

.PHONY: all check check-bootstrap clean
