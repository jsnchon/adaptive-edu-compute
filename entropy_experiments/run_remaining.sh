#!/usr/bin/env bash
set +e
set -x
cd ~/cs224n-final-project
source venv/bin/activate

echo "RUN 0"
python3 entropythresholdExperiement1_100.py --num_examples 500 --max_passes 5 --entropy_key p90_entropy_nats --threshold auto --threshold_mode low --threshold_source correct_median --output pipeline_results/lowEntropyp90_v1_noSkip.json --resume

echo "RUN 1"
python3 entropythresholdExperiement1_100.py --num_examples 500 --max_passes 5 --entropy_key mean_entropy_nats --threshold auto --threshold_mode high --threshold_source wrong_median --output pipeline_results/highEntropyAllowBoth.json --resume

echo "RUN 2"
python3 entropythresholdExperiement1_100.py --num_examples 500 --max_passes 5 --entropy_key mean_entropy_nats --threshold auto --threshold_mode low --threshold_source midpoint --output pipeline_results/entropythresholdExperiement1_100.json --resume

echo "RUN 3"
python3 entropythresholdExperiement1_100.py --num_examples 500 --max_passes 5 --entropy_key mean_entropy_nats --threshold auto --threshold_mode low --threshold_source correct_median --output pipeline_results/entropythresholdExperiement1_100_v2.json --resume

echo "RUN 4"
python3 entropythresholdExperiement2_100.py --num_examples 500 --max_passes 5 --entropy_key p90_entropy_nats --threshold auto --threshold_mode low --threshold_source correct_median --output pipeline_results/lowEntropyp90_v2_skipWrong.json --resume

echo "RUN 5"
python3 entropythresholdExperiement2_100.py --num_examples 500 --max_passes 5 --entropy_key mean_entropy_nats --threshold auto --threshold_mode high --threshold_source wrong_median --output pipeline_results/highEntropyExplanationsCorrect.json --resume
