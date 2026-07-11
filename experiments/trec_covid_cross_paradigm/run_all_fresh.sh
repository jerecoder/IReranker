#!/usr/bin/env bash
set -euo pipefail

python experiments/trec_covid_cross_paradigm/prepare_bm25.py
python experiments/trec_covid_cross_paradigm/run_ours_fresh.py --device cuda
bash experiments/trec_covid_cross_paradigm/run_baselines.sh
python experiments/trec_covid_cross_paradigm/build_metrics.py

tar -czf trec-covid-cross-paradigm-results.tar.gz \
  experiments/trec_covid_cross_paradigm/metrics \
  experiments/trec_covid_cross_paradigm/runs \
  experiments/trec_covid_cross_paradigm/logs

echo "Saved trec-covid-cross-paradigm-results.tar.gz"
