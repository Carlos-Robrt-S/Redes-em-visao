#!/usr/bin/env bash
# GPU 1: as quatro CNNs próprias (baseline + 3 variações arquiteturais).
# As épocas vêm de HYPER_PARAMS["EPOCHS"] em dataset/hyper_param.py.
source "$(dirname "$0")/_common.sh" cnn

run self_nn/metadata baseline   --model baseline
run self_nn/metadata batch_norm --model batch_norm
run self_nn/metadata wide       --model wide
run self_nn/metadata gap        --model gap

finish self_nn/results self_nn/metadata
