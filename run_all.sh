#!/usr/bin/env bash
# Rebuild experiment results and figures from the included data.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results report/figures

if [[ ! -f data/processed/dataset.csv ]]; then
    "$PYTHON" src/prepare_data.py
fi

for model in cnn lstm; do
    "$PYTHON" src/train.py --model "$model" --mode output --seed 0 --save_model
    for seed in 1 2; do
        "$PYTHON" src/train.py --model "$model" --mode output --seed "$seed"
    done
    for mode in input both; do
        "$PYTHON" src/train.py --model "$model" --mode "$mode"
    done
    for ablation in strip_format shuffle trunc64; do
        "$PYTHON" src/train.py --model "$model" --ablation "$ablation"
    done
    while read -r source target; do
        "$PYTHON" src/train.py --model "$model" --train_tasks "$source" --test_tasks "$target"
    done <<'TASKS'
coding,math,qa_other writing
qa_other coding
coding,math,writing qa_other
coding math
coding writing
math coding
math writing
writing coding
TASKS
done

"$PYTHON" src/baseline_bow.py
"$PYTHON" src/analysis_rq4.py
"$PYTHON" src/analysis_labels.py
"$PYTHON" src/make_figures.py
"$PYTHON" src/analysis_modern.py
printf '\nDone. Results are in results/; figures are in report/figures/.\n'
