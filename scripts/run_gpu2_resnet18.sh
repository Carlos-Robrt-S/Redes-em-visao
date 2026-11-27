#!/usr/bin/env bash
# GPU 2: ResNet18 em todas as estratégias. Obrigatórias primeiro, opcionais depois.
source "$(dirname "$0")/_common.sh" resnet18

run transfer_nn/metadata resnet18_frozen  --model resnet18 --strategy frozen
run transfer_nn/metadata resnet18_full    --model resnet18 --strategy full
run transfer_nn/metadata resnet18_partial --model resnet18 --strategy partial
run transfer_nn/metadata resnet18_scratch --model resnet18 --strategy scratch

finish transfer_nn/results/resnet18_* transfer_nn/metadata/resnet18_*
