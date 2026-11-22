# Validation-split rerun: 2026-11-22

Two fresh models, seed 42, 20 epochs, Adam (lr 0.001), batch size 64. Both use the same 55,000/5,000 train/validation split and initialization seed. Official test set: 10,000 images, evaluated once per validation-selected checkpoint.

| Activation | Selected epoch | Validation accuracy | Test correct | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Sigmoid | 17 | 98.58% | 9886/10000 | 98.86% |
| Relu | 17 | 99.30% | 9932/10000 | 99.32% |

ReLU scored 0.46 percentage points higher in this one-seed run. This is descriptive evidence, not a multi-seed statistical superiority claim. Training-only random rotation is applied to both models; validation and test receive no random augmentation.

## Provenance

- Training source SHA256: `610205d551b49ac461a4bfb58390e6ba4863c4f2afd3ea1c0463349fcb771cd0`.
- CPU run; exact Python/PyTorch/torchvision/platform/thread metadata and split/checkpoint hashes are recorded in each results JSON.
- Per-epoch train/validation metrics are stored in the same JSON files. Test metrics were obtained only after checkpoint selection.
- No prior committed checkpoint was loaded. Historical test-based-selection weights/plots remain unchanged outside this evaluation folder.
- The test split was used during historical coursework. The corrected split protocol should not be described as a historically unseen test set.
- Generated checkpoint and plot files remain in the ignored local run directory; rerun the documented command to regenerate them.
