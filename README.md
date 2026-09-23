# FVBA CIFAR-10 Experiments

This project implements the FVBA image-classification experiment pipeline for
controlled clean-label data-poisoning studies. Generated checkpoints are research artifacts and must
be treated as intentionally backdoored models.

## Pipeline components

- `fvba/models/lcng.py`: four-stage LCNG and bounded residual output.
- `fvba/models/adain.py`: class-conditioned adaptive instance normalization.
- `fvba/attacks/dct.py`: per-channel 8x8 block DCT.
- `fvba/attacks/losses.py`: CFAL and visual loss.
- `fvba/engine/generator.py`: FPBA and two VTBA training stages.
- `fvba/data/poison.py`: label-preserving poison construction.
- `fvba/evaluation/metrics.py`: strict non-target attack success metrics.

## Setup

Python 3.10+, PyTorch 2.1+, torchvision 0.16+, timm 1.0+, PyYAML and
pytest are required. Install the project in editable mode with
`python -m pip install -e '.[test,evaluation]'`. LPIPS is evaluation-only.

VTBA requires genuine ImageNet-21k ViT-B/16 weights. Set
`model.vit_checkpoint` to a local checkpoint or explicitly enable
`model.allow_weight_download`. Missing weights never fall back to a random
backbone.

## Commands

```bash
python -m fvba.cli prepare-data --config configs/cifar10_fpba.yaml
python -m fvba.cli run --config configs/cifar10_fpba.yaml
python -m fvba.cli run --config configs/cifar10_vtba.yaml
python -m fvba.cli run --config configs/cifar10_smoke.yaml
python -m fvba.cli run --config configs/cifar10_preflight.yaml
pytest -q
```

The full matrix launcher is `scripts/run_cifar10_main.sh`. Full runs require
CUDA unless `runtime.allow_cpu_full_run` is explicitly enabled.

For CUDA data loading, `pin_memory` and `persistent_workers` can be enabled in
the data configuration. Distributed execution is available through the
`fvba.distributed` context and should be enabled only with `torchrun`.

## Configuration notes

Batch size, epoch counts, weight decay, convolution details, augmentation, and
the perturbation parameterization are explicit in each configuration. The
default implementation uses a 4x4 stride-2 LCNG, residual `epsilon*tanh`,
CIFAR crop/flip augmentation, 150 generator epochs, 100 victim epochs, batch
size 128, and victim weight decay 5e-4.
