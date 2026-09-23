#!/usr/bin/env bash
set -euo pipefail

for method in fpba vtba; do
  for victim in resnet18 vgg19; do
    for seed in 0 1 2; do
      python -m fvba.cli run \
        --config "configs/cifar10_${method}.yaml" \
        --victim "${victim}" \
        --seed "${seed}" \
        --output-dir "outputs/cifar10_${method}_${victim}_seed${seed}"
    done
  done
done

