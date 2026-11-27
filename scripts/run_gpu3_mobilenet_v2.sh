#!/usr/bin/env bash
# GPU 3: MobileNetV2 em todas as estratégias. Obrigatórias primeiro, opcionais depois.
source "$(dirname "$0")/_common.sh" mobilenet_v2

run transfer_nn/metadata mobilenet_v2_frozen  --model mobilenet_v2 --strategy frozen
run transfer_nn/metadata mobilenet_v2_full    --model mobilenet_v2 --strategy full
run transfer_nn/metadata mobilenet_v2_partial --model mobilenet_v2 --strategy partial
run transfer_nn/metadata mobilenet_v2_scratch --model mobilenet_v2 --strategy scratch

finish transfer_nn/results/mobilenet_v2_* transfer_nn/metadata/mobilenet_v2_*
