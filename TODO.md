# TODO

The 30-epoch run on CIFAR-10 shows no collapse, but the predictions only capture local colours and textures (see [PROGRESS.md](PROGRESS.md)). The next iteration has two goals:

1. re-train on a higher-resolution dataset;
2. increase the dimension of the embedding space.

Make the two changes **one at a time**, so the effect of each can be measured separately: first the larger dimension on CIFAR-10 (run `cifar10_d256`, compared with the `cifar10_d128` baseline), then the dataset.

---

## 1. Higher-resolution dataset

**Choice: STL-10 resized to 64x64**, available through `tensorflow_datasets`:
- `unlabelled` split (100k images) for pre-training, which is what it was built for;
- labelled `train` (5k) / `test` (8k) splits for the probe and the analysis scripts.

Tiny ImageNet (64x64) is the alternative, but it is not in `tfds` as standard and has to be downloaded manually.

**Patch size: 8 instead of 4.** At 64x64 this keeps an 8x8 grid of 64 patches, so the masking, the tests and the cost of the attention stay the same, and each patch carries more detail. Keeping patches of 4 would give 256 tokens and an attention about 16x more expensive.

### Data pipeline: `jepa/data.py`
- [x] Add a `dataset` field to `JEPAConfig` (`"cifar10"` | `"stl10"`); `DATASETS` in `jepa/config.py` sets `image_size`, `patch_size`, `mean` and `std` for each dataset.
- [x] `load_split()` dispatches on `config.dataset`. For STL-10: load with `tfds`, resize 96 -> 64 (bilinear, antialiased, back to uint8).
- [x] `build_pretrain_dataset`: the `unlabelled` split for STL-10 (781 steps per epoch), streamed from `tfds` (< 0.8 GB of RAM for the pipeline).
- [x] `build_probe_dataset`: labelled `train` / `test` splits for STL-10.
- [x] Compute the per-channel mean / std of STL-10 (unlabelled split, resized to 64x64) and store them in the config: mean (0.4406, 0.4273, 0.3858), std (0.2592, 0.2519, 0.2597).
- [x] `augment`: the reflect padding is `image_size // 8` (4 px at 32x32, 8 px at 64x64).
- [x] Keep `CIFAR10_CLASSES` and add `STL10_CLASSES` (`frog` is replaced by `monkey`).

### Training: `jepa/train.py`
- [x] Update the module docstring (it says "on CIFAR-10").
- [x] Choose the dataset: through `--run`, the presets carry the dataset.
- [x] Write checkpoints and logs to a folder per run so the first run is not overwritten: `--run <preset>` (presets in `jepa/config.py`), weights in `checkpoints/<run>/`, log in `logs/<run>.log`. The first run was moved to `checkpoints/cifar10_d128/` and `logs/cifar10_d128.log`.
- [x] Add the STL-10 presets to `PRESETS` in `jepa/config.py`: `stl10_d128`, `stl10_d256`.
- [ ] Re-check the batch size and the number of epochs: 100k images instead of 50k, so twice as many steps per epoch.

### Analysis scripts
- [x] `result_analysis/check_collapse.py`, `result_analysis/predicted_fill.py`, `experimentation/test_image.py`: `--run` loads the matching config and checkpoint.
- [x] `result_analysis/predicted_fill.py`: update the "CIFAR-10" docstrings and labels; the retrieval bank is the test split of the run's dataset.
- [x] `result_analysis/plot_loss.py`: `--run` reads `logs/<run>.log`.
- [x] `experimentation/test_image.py`: update the comments and plot titles that mention 32x32 / CIFAR-10.
- [ ] Once trained, re-run on `img/bird1.jpg` (`prediction_bird1_v2.png`) to compare with v1.
- [x] `jepa/visualise.py`: `--run`, masks plot checked at 64x64.

### Tests
- [x] Parametrise `tests/test_data.py` on both datasets (shapes, standardisation, augmentation, 96x96 before the resize, unlabelled split).
- [x] Every preset runs through the full model, and both datasets keep the 8x8 grid (`tests/test_config.py`).

### Docstrings
- [x] Update the shape comments written for 32x32 and 4x4 patches (`jepa/patch_embed.py`, `jepa/model.py`).

---

## 2. Larger embedding dimension

Mostly a config change in `jepa/config.py`:
- [x] `embed_dim`: 128 -> 256.
- [x] `predictor_embed_dim`: 64 -> 128 (keep the predictor narrower than the encoder, as in I-JEPA).
- [x] Keep `embed_dim` divisible by `num_heads` (and by 4, for the sin-cos position embedding): `num_heads` 4 -> 8 to keep 32 dims per head, `predictor_num_heads` 4 -> 8 to keep 16 dims per head (`tests/test_config.py`).
- [x] Update the shape comments written for `embed_dim=128` (`jepa/attention.py`, `jepa/model.py`, `jepa/patch_embed.py`).
- [x] Check the tests that hard-code 128 / 64.
- [ ] Watch memory and training time: the parameter count grows about 4x (0.92M -> 3.63M trainable), the baseline took ~4 h on CPU.
- [ ] Train `cifar10_d256` on a Kaggle GPU (see section 3 below), or locally: `python -m jepa.train --run cifar10_d256 > logs/cifar10_d256.log 2>&1`.

---

## 3. Training on a cloud GPU (Kaggle)

The CPU is too slow for the new runs (the baseline already took ~4 h). TensorFlow has had no GPU support on native Windows since 2.11, so the runs go to a free Kaggle GPU (T4 or P100, ~30 h per week).

- [x] `notebooks/kaggle_train.ipynb`: clones the repo, checks that a GPU is visible, runs a 20-step smoke test, then trains the `RUN` preset. It keeps Kaggle's own TensorFlow, writes `checkpoints/<run>/` and `logs/<run>.log`, and zips them into `<run>.zip`, to unzip at the project root.
- [ ] Import the notebook into Kaggle (Accelerator: GPU, Internet: on) and run `cifar10_d256` with *Save & Run All*.
- [ ] Note the time per step in the smoke test. `train_step` runs eagerly (no `tf.function`), so the GPU speed-up may be small. If it is, compile `train_step`, keeping in mind that the mask sizes change from one batch to the next and would cause retracing.
- [ ] There is no resume from a checkpoint: a session that stops (12 h limit) means restarting the run. Add a resume (model and optimizer state, step, RNG) if the STL-10 runs come close to the limit.

---

## 4. Evaluation and comparison

For each run (`cifar10_d128` as the baseline, `cifar10_d256`, then STL-10):
- [ ] Loss curve (`plot_loss.py`).
- [ ] Collapse check: normalised std (ideal ~`1/sqrt(embed_dim)`), dead dimensions, effective rank.
- [ ] Nearest-neighbour retrieval rate of the true patch (`predicted_fill.py`), currently 51%.
- [ ] Qualitative test on `experimentation/img/bird1.jpg`, currently ~46%: is the shape of the bird / branch recovered?
- [ ] Optional: a linear probe on the labelled splits (`build_probe_dataset` already exists) for a quantitative comparison of the representations.
- [ ] Add the results to `result_analysis/RESULTATS.md` and update `PROGRESS.md`.
