# lp_tokenizer Makefile
# Targets: install, smoke, run, eval, clean

.PHONY: install smoke run eval clean test

# ── Install Python dependencies ───────────────────────────────
install:
	pip install -r requirements.txt

# ── Quick smoke test (~5 min on 4090) ────────────────────────
smoke:
	./experiments/run_experiment.sh smoke

# ── Full experiment (~2-4h on 4090) ──────────────────────────
run:
	./experiments/run_experiment.sh

# ── Download data only ────────────────────────────────────────
data:
	python3 data/download.py --dataset tinystories

# ── Tokenizer metrics only (requires trained tokenizer) ───────
eval-tokenizer:
	python3 eval/tokenizer_metrics.py \
		--text_file data/raw/tinystories/val_0000.txt \
		--clustering_tokenizer data/clustering/tokenizer.bin

# ── Generation quality comparison (requires training logs) ────
eval-generation:
	python3 eval/generation_metrics.py \
		--mode logs \
		--bpe_log experiments/out/bpe/log.txt \
		--cls_log experiments/out/clustering/log.txt

# ── Run unit tests ────────────────────────────────────────────
test:
	python3 -m pytest tests/ -v 2>/dev/null || python3 tokenizer/clustering_tokenizer.py
	python3 tokenizer/tokenizer_utils.py
	python3 tokenizer/bpe_baseline.py

# ── Clean generated data (keep raw downloads) ─────────────────
clean:
	rm -rf data/bpe/ data/clustering/ experiments/out/

# ── Clean everything including raw data ───────────────────────
distclean: clean
	rm -rf data/raw/
