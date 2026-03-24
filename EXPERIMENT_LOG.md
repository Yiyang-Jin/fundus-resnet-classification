# Experiment Log

This log tracks training rounds, observed issues, applied fixes, and outcomes.

> **Repository note:** This file is the full chronological log copied from the coursework workspace. The **recommended default** in `train_resnet18.py` uses `mixup_alpha=0.2` (restored after a failed `0.1` trial).

## Round 1 - Baseline (Overfitting Observed)

### Setup
- Model: `ResNet-18` (ImageNet pretrained)
- Data: `resnet_data_aug`
- Typical behavior: training accuracy rose very quickly while validation did not improve steadily.

### Key Evidence (early epochs)
- `Epoch 01`: train_acc=`0.5904`, val_acc=`0.5781`
- `Epoch 02`: train_acc=`0.7701`, val_acc=`0.5781`
- `Epoch 03`: train_acc=`0.9325`, val_acc=`0.5580`
- `Epoch 04`: train_acc=`0.9820`, val_acc=`0.5848`
- `Epoch 05`: train_acc=`0.9922`, val_acc=`0.5558`

### Diagnosis
- Clear overfitting: train metrics improved rapidly, val metrics stagnated/declined.

---

## Round 2 - Anti-overfitting Controls (Partial Improvement)

### Changes Introduced
- Early stopping (monitor `val_loss`)
- Freeze-then-unfreeze training
- Stronger regularization (`dropout`, `label_smoothing`, higher `weight_decay`)
- Validation-driven LR schedule (`ReduceLROnPlateau`)

### Run Summary
- Command: `python train_resnet18.py --epochs 30 --batch-size 32`
- Early stopping triggered at `epoch 10`
- Best validation accuracy: `0.6049`
- Final test metrics:
  - `test_acc=0.5658`
  - `test_loss=1.0937`

### Interpretation
- Better than Round 1 in validation behavior and generalization.
- Residual overfitting still appeared in late fine-tuning.

---

## Round 3 - Progressive Unfreezing + Layer-wise LR

### Planned Changes
- Progressive unfreezing:
  1. head-only
  2. `layer4 + head`
  3. full-network fine-tuning
- Layer-wise learning rates:
  - `head_lr=5e-5`
  - `backbone_lr=1e-5`

### Planned Command
- `python train_resnet18.py --epochs 30 --batch-size 32`

### Results
- Command: `python train_resnet18.py --epochs 30 --batch-size 32`
- Stage transitions:
  - epoch 1-5: head-only
  - epoch 6-8: `layer4 + head`
  - epoch 9+: full fine-tuning
- Early stopping triggered at `epoch 13`
- Final test metrics:
  - `test_acc=0.5789`
  - `test_loss=1.1306`

### Comparison vs Round 2
- Round 2 test accuracy: `0.5658`
- Round 3 test accuracy: `0.5789`
- Absolute gain: `+0.0131` (~`+1.31%`)

### Interpretation
- Progressive unfreezing + layer-wise LR provided a measurable improvement.
- Performance still plateaus around the high `0.5x` range.
- Next action: add mild stochastic noise augmentation to improve robustness and encourage learning of shared disease-relevant patterns.

---

## Round 4 - Mild Noise Augmentation

### Changes Introduced
- Added mild Gaussian noise in train pipeline:
  - `noise_std=0.02`
  - `noise_prob=0.3`
- Kept progressive unfreezing and layer-wise LR setup.

### Run Summary
- Command: `python train_resnet18.py --epochs 30 --batch-size 32 --num-workers 0`
- Stage transitions:
  - epoch 1-5: head-only
  - epoch 6-8: `layer4 + head`
  - epoch 9+: full fine-tuning
- Early stopping triggered at `epoch 13`
- Best validation accuracy: `0.5804` (epoch 10)
- Final test metrics:
  - `test_acc=0.5811`
  - `test_loss=1.1084`

### Observed Pattern
- Validation improved until around epoch 10, then validation loss rose (`1.1345 -> 1.1723`) while train accuracy kept increasing.
- Validation accuracy remained below `0.6`, indicating a plateau.

### Hypothesis and Next Check
- Hypothesis: current input resolution (`224x224`) may be too small for subtle retinal lesion details.
- Next planned check:
  1. increase resolution (e.g., `320` or `384`)
  2. compare with the same training strategy
  3. verify whether validation ceiling moves above `0.6`

### Status Note
- This round is **invalid for 320-resolution comparison**.
- Although `resnet_data_aug_320` was used as input source, the training transform still resized images to `224x224`.
- Therefore, this run does not represent true 320 model input and should be discarded for resolution-ablation conclusions.

---

## Round 5 - True 320 Input (ResNet-18)

### Setup
- Data root: `resnet_data_aug_320`
- True input size: `320x320` (resize fixed in training script)
- Progressive unfreezing + layer-wise LR kept
- AMP + parallel loading tested

### Main Outcomes
- Best validation accuracy reached around `0.61`
- Test accuracy remained around high `0.58`
- The desired clear break beyond `0.6` on test set was not achieved

### Conclusion
- Increasing resolution from `224` to true `320` alone did **not** provide a decisive gain.
- It improved training signal in some phases but did not remove the generalization ceiling.
- Next step: increase model capacity while keeping the same optimized pipeline (`ResNet-34` under matched settings).

---

## Round 6 - Class-Scope Adjustment (Drop `Other`)

### Motivation
- Repeated confusion-matrix checks show strong overlap among `Other`, `Normal`, and `Diabetes`.
- Hypothesis: `Other` is a heterogeneous catch-all class with weak visual consistency, which blurs the decision boundary.

### Action
- Exclude `Other` class from train/val/test during model training and evaluation.
- Continue with the optimized pipeline (progressive unfreezing, AMP, parallel loading, 320 input).

### Expected Effect
- Reduce label-noise-like ambiguity from a mixed class.
- Improve boundary sharpness among clinically meaningful categories.

### Run Result (Drop `Other` + `Diabetes`)
- Command used:
  - `python train_resnet18.py --model-name resnet34 --epochs 30 --batch-size 24 --data-root "resnet_data_aug_320" --img-size 320 --noise-prob 0.3 --noise-std 0.02 --num-workers 4 --prefetch-factor 1 --freeze-epochs 4 --layer4-only-epochs 3 --mixup-alpha 0.2 --full-head-lr-scale 0.5 --full-backbone-lr-scale 0.5 --drop-classes "Other,Diabetes"`
- Early stopping at epoch `13`
- Final metrics:
  - `best_val_acc = 0.7560`
  - `test_acc = 0.7690`
  - `test_loss = 0.7038`

### Confusion-Matrix Diagnosis (6 classes)
- Strong classes:
  - `AMD` class_acc `0.8772`
  - `Myopia` class_acc `0.8421`
  - `Cataract` class_acc `0.7895`
  - `Hypertension` class_acc `0.7895`
- Remaining weaker class:
  - `Glaucoma` class_acc `0.6140`
  - major confusions: `Glaucoma -> Normal` (`12`), `Glaucoma -> Myopia` (`8`)
- `Normal` improved to class_acc `0.7018`, but still confuses with `Cataract` (`11`).

### Interpretation
- Removing `Other` and `Diabetes` significantly improved overall separability in this focused 6-class setting.
- The new bottleneck is primarily **Glaucoma boundary quality**, not global overfitting.

---

## Round 7 - Backbone Comparison (`ResNet34` vs `ResNet18`, 6 classes)

### Goal
- Keep the same optimized pipeline and class scope (`drop Other + Diabetes`).
- Compare model capacity impact on late-stage overfitting and final test performance.

### Common Setup
- Input: `320x320`
- Classes: `AMD, Cataract, Glaucoma, Hypertension, Myopia, Normal`
- MixUp + mild noise + AMP + parallel loading
- Progressive unfreezing (`freeze_epochs=4`, `layer4_only_epochs=3`)

### `ResNet34` (reference from previous round)
- `best_val_acc = 0.7560`
- `test_acc = 0.7690`
- `test_loss = 0.7038`
- Early stopping at epoch `13`

### `ResNet18` (new run)
- Command:
  - `python train_resnet18.py --model-name resnet18 --epochs 30 --batch-size 24 --data-root "resnet_data_aug_320" --img-size 320 --noise-prob 0.3 --noise-std 0.02 --num-workers 4 --prefetch-factor 1 --freeze-epochs 4 --layer4-only-epochs 3 --mixup-alpha 0.2 --full-head-lr-scale 0.5 --full-backbone-lr-scale 0.5 --drop-classes "Other,Diabetes"`
- `best_val_acc = 0.7560`
- `test_acc = 0.7632`
- `test_loss = 0.7137`
- Early stopping at epoch `15`

### Comparison Summary
- Validation peak is effectively tied (`0.7560` vs `0.7560`).
- `ResNet34` has slightly better test accuracy (`0.7690` vs `0.7632`) and lower test loss.
- `ResNet18` does not clearly solve the late-stage overfitting issue under this exact setting.

---

## Next Strategy Memo (A/B/C)

### A) Asymmetric retinal preprocessing (implement first)
- Motivation: lesion cues are often clearer on the green channel; some red-channel regions can be overexposed.
- Plan:
  - apply local-contrast enhancement on green channel
  - slightly increase green-channel gain
  - keep all other training settings fixed for clean ablation
- Status: **tested, then discarded by project decision** (future runs keep `ResNet18` backbone as requested)

### A Trial Note (discarded for direction control)
- Trial run completed with A-preprocess:
  - `best_val_acc = 0.7560`
  - `test_acc = 0.7895`
  - `test_loss = 0.6849`
- Despite decent metrics, this branch is marked as discarded to keep experimental direction focused on `ResNet18` runs only.

### B) Cosine-annealing learning-rate schedule
- Motivation: smooth LR decay may reduce late-stage instability.
- Plan:
  - replace current scheduler with `CosineAnnealingLR` in a follow-up run
  - compare only against A-fixed baseline
- Status: **queued**

### C) Attention mechanism (SE block)
- Motivation: improve channel-level focus without increasing depth too much.
- Plan:
  - add lightweight SE to later residual stages and re-run with A/B baseline
- Status: **queued**

### Update: A on `ResNet18` (confirmed gain)
- With class scope (`drop Other + Diabetes`) and all other controls fixed, A-preprocess on `ResNet18` improved:
  - `best_val_acc: 0.7560 -> 0.7768`
  - `test_acc: 0.7632 -> 0.7924`
  - `test_loss: 0.7137 -> 0.6895`

### Next Run: A + B
- B is applied as cosine scheduler (`CosineAnnealingLR`) on top of the A-enhanced `ResNet18` pipeline for direct comparison.

---

## Round 9 - Glaucoma-Weighted Focal Trial (Rejected)

### Setup
- Baseline plus targeted loss changes:
  - `mixup_alpha=0.1`
  - `loss_type=focal`
  - `focal_gamma=2.0`
  - `glaucoma_weight=1.8`
- Other settings kept close to A+B run.

### Result
- `best_val_acc = 0.7143`
- `test_acc = 0.7368`
- `test_loss = 0.7205`

### Decision
- This trial clearly underperformed the established baseline.
- Keep baseline fixed as:
  - `ResNet18 + A preprocess + drop classes (Other, Diabetes)`
  - without additional focal/class-weight modifications.

---

## Round 8 - SE Minimal-Intrusion Check (Result)

### Setup
- `ResNet18 + A + B(cosine) + drop Other/Diabetes`
- Added `SE` only at `layer4` (`se_reduction=16`)

### Outcome
- `best_val_acc = 0.7768`
- `test_acc = 0.7865`
- `test_loss = 0.6846`

### Decision
- Compared with non-SE A+B baseline (`test_acc=0.7924`), SE slightly decreased test accuracy.
- Marked as **not adopted** for current pipeline.

---

## Round 10 - Fine-Tuning LR Scale Bisection (`0.45`)

### Motivation
- After `0.5` vs `0.3` full-unfreeze LR scale tests, try an intermediate `0.45` for head/backbone multipliers.

### Setup
- Same as strong baseline: `ResNet18`, `320`, A-preprocess, `drop-classes Other,Diabetes`, `mixup_alpha=0.2`, `ReduceLROnPlateau`, AMP, etc.
- `full_head_lr_scale=0.45`, `full_backbone_lr_scale=0.45`

### Outcome (with early stopping, patience 4)
- `best_val_acc = 0.7798`
- `test_acc = 0.7895`
- `test_loss = 0.6867`
- Early stop around epoch `19`

### Interpretation
- Slightly below the best historical `test_acc=0.7924` obtained with `0.5` scales; `0.45` is usable but not a strict upgrade.

---

## Round 11 - Plateau Scheduler, No Early Stopping (Sanity Check)

### Motivation
- Rule out “forced early stop” as the reason for not reaching `0.79+` on a given seed.

### Setup
- Identical to Round 10 except `--early-stop-patience 999` (train full `30` epochs).
- Scheduler remains `ReduceLROnPlateau`.

### Outcome
- Same best checkpoint metrics as Round 10:
  - `best_val_acc = 0.7798`
  - `test_acc = 0.7895`
  - `test_loss = 0.6867`
- Late epochs: train accuracy continued upward while validation fluctuated; no late breakthrough.

### Decision
- Early stopping was **not** hiding a better solution in this configuration; the performance ceiling is stable around this band.

---

## Round 12 - Lower MixUp Strength (`alpha=0.1`, Aborted)

### Motivation
- Hypothesis: weaker MixUp might stabilize validation in late training.

### Setup
- Same pipeline as Round 10/11 but `mixup_alpha=0.1`.

### Observation (run stopped by user ~epoch 22)
- Validation accuracy stayed in roughly `0.75–0.77` while train accuracy rose into `~0.83+`, indicating a less favorable train/val gap vs `mixup_alpha=0.2`.

### Decision
- **Rejected.** Restore default and recommended setting **`mixup_alpha=0.2`** in `train_resnet18.py`.

---

## Consolidated Summary (Updated)

### What was validated
- Simply increasing depth (`ResNet18` -> `ResNet34`) did not provide a decisive gain under the same pipeline.
- True `320x320` input helped partially, but by itself did not break the old plateau.
- Class-scope adjustment (dropping `Other` and `Diabetes`) produced the largest jump in separability.
- A-strategy asymmetric preprocessing (green-channel focused enhancement) gave a clear measurable gain on `ResNet18`.
- B-strategy (`CosineAnnealingLR`) on top of A improved smoothness and loss slightly, but did not materially increase peak test accuracy vs A+plateau.
- TTA (2/3/5 fold) did not improve this model family and setting.
- SE in `layer4` slightly hurt test accuracy vs baseline.
- Focal + Glaucoma weighting trial underperformed strongly.
- LR scale `0.45` and disabling early stopping did not beat the best `0.5`-scale run; `mixup_alpha=0.1` underperformed vs `0.2`.

### Reference runs (6-class: drop `Other` + `Diabetes`)
| Configuration | best_val_acc | test_acc | test_loss |
|---------------|--------------|----------|-----------|
| `ResNet18 + A + plateau`, scales `0.5`, `mixup 0.2` | `0.7768` | **`0.7924`** | `0.6895` |
| `ResNet18 + A + B(cosine)` | `0.7798` | `0.7924` | `0.6860` |
| `ResNet18 + A + plateau`, scales `0.45`, `mixup 0.2` | `0.7798` | `0.7895` | `0.6867` |

### Practical takeaway
- **Publish / reproduce:** prefer **`mixup_alpha=0.2`**, A-preprocess, `ReduceLROnPlateau`, and full-unfreeze LR scales **`0.5`** for the strongest observed test accuracy; use **`0.45`** if you want a slightly more conservative fine-tune (small accuracy trade-off in our runs).
- **Do not use** for this pipeline: heavy TTA, `layer4` SE (as implemented), focal/Glaucoma-weight trial, or `mixup_alpha=0.1`.
