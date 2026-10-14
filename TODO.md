# TODO

The 30-epoch run on CIFAR-10 shows no collapse, but the predictions only capture local colours and textures (see [PROGRESS.md](PROGRESS.md)). The next iteration has two goals:

1. re-train on a higher-resolution dataset;
2. increase the dimension of the embedding space.

Make the two changes **one at a time**, so the effect of each can be measured separately: first the dataset with `embed_dim=128`, then the larger dimension.

---

## 1. Higher-resolution dataset

**Choice: STL-10 resized to 64x64**, available through `tensorflow_datasets`:
- `unlabelled` split (100k images) for pre-training, which is what it was built for;
- labelled `train` (5k) / `test` (8k) splits for the probe and the analysis scripts.

Tiny ImageNet (64x64) is the alternative, but it is not in `tfds` as standard and has to be downloaded manually.

**Patch size: 8 instead of 4.** At 64x64 this keeps an 8x8 grid of 64 patches, so the masking, the tests and the cost of the attention stay the same, and each patch carries more detail. Keeping patches of 4 would give 256 tokens and an attention about 16x more expensive.

### Data pipeline: `jepa/data.py`
- [ ] Add a `dataset` field to `JEPAConfig` (`"cifar10"` | `"stl10"`), and a helper or preset that sets `image_size`, `patch_size`, `mean` and `std` for each dataset.
- [ ] Replace `load_cifar10()` with a loader that dispatches on `config.dataset`. For STL-10: load with `tfds`, resize 96 -> 64.
- [ ] `build_pretrain_dataset`: use the `unlabelled` split for STL-10. Do not load everything in memory with `from_tensor_slices` if it does not fit, stream from `tfds` instead.
- [ ] `build_probe_dataset`: labelled `train` / `test` splits for STL-10.
- [ ] Compute the per-channel mean / std of STL-10 (on the resized images) and store them in the config.
- [ ] `augment`: the reflect padding is hard-coded to 4 px (written for 32 px). Make it proportional to `image_size` (e.g. `image_size // 8`).
- [ ] Keep `CIFAR10_CLASSES` and add the STL-10 class names.

### Training: `jepa/train.py`
- [ ] Update the module docstring (it says "on CIFAR-10").
- [ ] Add a `--dataset` argument.
- [ ] Write checkpoints and logs to a folder per run (e.g. `checkpoints/stl10_d128/`, `logs/stl10_d128.log`) so the CIFAR-10 run (`checkpoints/jepa.weights.h5`, `logs/train.log`) is not overwritten.
- [ ] Re-check the batch size and the number of epochs: 100k images instead of 50k, so twice as many steps per epoch.

### Analysis scripts
- [ ] `result_analysis/check_collapse.py`, `result_analysis/predicted_fill.py`: the checkpoint path is hard-coded (`checkpoints/jepa.weights.h5`). Add an argument for the checkpoint / run, and pass the matching config.
- [ ] `result_analysis/predicted_fill.py`: update the "CIFAR-10" docstrings and labels; the retrieval bank becomes the STL-10 test patches.
- [ ] `result_analysis/plot_loss.py`: take the log file as an argument.
- [ ] `experimentation/test_image.py`: the image is already resized to `config.image_size`; update the comments and plot titles that mention 32x32 / CIFAR-10. Re-run on `img/bird1.jpg` (`prediction_bird1_v2.png`) to compare with v1.
- [ ] `jepa/visualise.py`: check the masks plot at 64x64.

### Tests
- [ ] Parametrise `tests/test_data.py` on both datasets (shapes, standardisation, augmentation).
- [ ] Check the other tests that assume 32x32 / 64 patches (`test_patching.py`, `test_masking.py`, `test_model.py`, `test_visualise.py`) still pass with `image_size=64, patch_size=8`.

### Docstrings
- [ ] Update the shape comments written for 32x32 and 4x4 patches (`jepa/patch_embed.py`, `jepa/model.py`).

---

## 2. Larger embedding dimension

Mostly a config change in `jepa/config.py`:
- [ ] `embed_dim`: 128 -> 256.
- [ ] `predictor_embed_dim`: 64 -> 128 (keep the predictor narrower than the encoder, as in I-JEPA).
- [ ] Keep `embed_dim` divisible by `num_heads` (and even, for the sin-cos position embedding). Consider `num_heads` 4 -> 8 to keep 32 dims per head.
- [ ] Update the shape comments written for `embed_dim=128` (`jepa/attention.py`, `jepa/model.py`, `jepa/patch_embed.py`).
- [ ] Check the tests that hard-code 128 / 64.
- [ ] Watch memory and training time: the parameter count grows roughly 4x in the encoder.

---

## 3. Evaluation and comparison

For each run (CIFAR-10 d=128 as the baseline, STL-10 d=128, STL-10 d=256):
- [ ] Loss curve (`plot_loss.py`).
- [ ] Collapse check: normalised std (ideal ~`1/sqrt(embed_dim)`), dead dimensions, effective rank.
- [ ] Nearest-neighbour retrieval rate of the true patch (`predicted_fill.py`), currently 51%.
- [ ] Qualitative test on `experimentation/img/bird1.jpg`, currently ~46%: is the shape of the bird / branch recovered?
- [ ] Optional: a linear probe on the labelled splits (`build_probe_dataset` already exists) for a quantitative comparison of the representations.
- [ ] Add the results to `result_analysis/RESULTATS.md` and update `PROGRESS.md`.
