# LeNet-5: Sigmoid vs. ReLU on MNIST

A course implementation of LeNet-5 with configurable Sigmoid/ReLU activations, PyTorch training, and a PyQt5 digit-classification interface.

## Evaluation protocol

The revised `train.py` retrains both models from scratch:

- Split MNIST's 60,000 official training images into **55,000 training / 5,000 validation** examples using seed 42. Both activations use identical split indices, initial parameter seeds, and loader seeds.
- Resize to 32 x 32 and normalize with mean/std 0.5. Apply random rotations of up to 15 degrees **only to training images**; validation and test preprocessing are deterministic.
- Train each activation for 20 epochs with Adam, learning rate 0.001 and batch size 64.
- Select the checkpoint with highest validation accuracy (earliest epoch on ties).
- Evaluate that checkpoint once on all **10,000 official test images**, after training and selection are finished. Test scores are not used to select an epoch.
- Record per-epoch loss and accuracy, split hashes, checkpoint hash, package versions and final test counts in JSON.

This is a single-seed descriptive comparison. It does not establish that one activation is statistically superior or generalize to other datasets. Historical work in this repository already exposed the test split during model selection; retraining fixes the current procedure but does not make the test set historically unseen.

## Recorded rerun (2026-11-22)

| Activation | Selected epoch | Validation accuracy | Test accuracy (10,000 images) |
| --- | ---: | ---: | ---: |
| Sigmoid | 17 | 98.58% | **98.86%** (9,886 correct) |
| ReLU | 17 | 99.30% | **99.32%** (9,932 correct) |

One seed per activation. Both models were retrained from scratch on CPU for 20 epochs. See [the report](evaluation/REPORT.md) and the per-epoch JSON records in [evaluation/](evaluation/). The 0.46-percentage-point difference is descriptive, not evidence of general superiority.

## Run

From the repository root, create an isolated Python environment and install the dependencies:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision matplotlib
python HW2/Hw2_1/train.py --seed 42 --epochs 20
```

`requirements-evaluation.txt` records the packages from the measured rerun environment; hardware-specific torch wheels may require platform-specific installation.

By default CUDA is used if available, otherwise CPU. `--device mps` is opt-in for Apple Silicon. Use a fresh `--output-dir` for each run; existing results are never overwritten. The default is `HW2/Hw2_1/runs/validation_split/`. Dataset downloads and new checkpoints are ignored by Git.

Generated artifacts include `results_Sigmoid.json`, `results_Relu.json`, `summary.json`, separate model weights and training/validation curves. Full metrics are retained in `evaluation/` after a documented completed run.

## Historical GUI artifacts

`main.py` implements the original GUI: architecture display, learning curves, digit prediction and a class-probability histogram. It loads `Weight_Relu.pth` and the plots from its working directory. The original root-level weights/plots are retained unchanged and come from the old test-based-selection protocol. To demonstrate a corrected checkpoint, run the GUI from the new output directory or supply copies of the new weights/plots there; install the GUI's additional dependencies (`opencv-python`, `PyQt5`, `torchsummary`). GUI behavior has not been revalidated as part of the evaluation rerun.
