# BridgeData V2 Adaptation Plan

This project currently completes the novel-dataset path for **DROID**. The
second target dataset, **BridgeData V2**, should be handled by converting it into
the same visual-token format used by 1XGPT and DROID.

The goal is not to map BridgeData actions into the original 1X action/state
format. Instead, we keep the adaptation **visual-only**:

```python
WITH_ACT = False
```

This avoids mixing incompatible action spaces across robot platforms.

---

## Required Output Contract

The Masked-HWM loader expects a split folder shaped like:

```text
train_v2.0/
  metadata.json
  metadata/
    metadata_0.json
  videos/
    video_0.bin
  segment_indices/
    segment_idx_0.bin

val_v2.0/
  metadata.json
  metadata_0.json
  video_0.bin
  segment_idx_0.bin
```

The validation split is flat because `RawTokenDataset(..., is_eval=True)` reads
`metadata_{rank}.json`, `video_{rank}.bin`, and `segment_idx_{rank}.bin`
directly from the eval directory.

Each tokenized video chunk should have shape:

```text
(3, 32, 32)
```

where the three latent timesteps decode to a 17-frame RGB clip with Cosmos
`DV8x8x8`.

---

## Proposed BridgeData V2 Pipeline

1. **Load BridgeData V2 trajectories**

   BridgeData V2 can appear as TFDS/RLDS-style episodes or as downloaded
   trajectories/videos depending on the source. The preprocessing notebook should
   support both:

   - RLDS episodes with `steps`;
   - raw videos or frame folders as fallback.

   This is now implemented in
   [`notebooks/bridge-v2-preprocessing.ipynb`](../notebooks/bridge-v2-preprocessing.ipynb),
   separate from the DROID preprocessing notebook.

2. **Select camera stream**

   Prefer a single consistent camera across all trajectories:

   - external camera for global scene context; or
   - wrist camera for manipulation detail.

   Do not mix cameras within the same split unless explicitly studying
   multi-view robustness.

3. **Normalize frames**

   Convert frames to RGB `uint8`, then resize/crop to the tokenizer input size
   used by the Cosmos checkpoint.

4. **Create 17-frame clips**

   Use the same temporal contract as the current Masked-HWM notebooks:

   ```text
   17 RGB frames -> Cosmos encoder -> 3 latent timesteps
   ```

5. **Encode with Cosmos**

   Use the uploaded Cosmos tokenizer files:

   ```text
   Cosmos-0.1-Tokenizer-DV8x8x8/
     encoder.jit
     decoder.jit
   ```

   The output token chunk should be `int32` with shape `(3,32,32)`.

6. **Write 1X-compatible binary shards**

   Save training token chunks into `videos/video_*.bin`; write matching
   `metadata/metadata_*.json`, `metadata.json`, and
   `segment_indices/segment_idx_*.bin`. Save validation shards in the flat eval
   layout used by `RawTokenDataset(..., is_eval=True)`.

7. **Verify compatibility**

   Reuse the same verification cell from the DROID preprocessing notebook:

   - split folders exist;
   - metadata and shard sizes match;
   - token min/max are within the codebook range;
   - `RawTokenDataset` can read train and validation examples.

8. **Run adaptation experiments**

   Recommended sequence:

   - zero-shot inference from the 1X-trained checkpoint;
   - bounded BridgeData fine-tuning, e.g. 500-1000 steps first;
   - paired PSNR/MSE comparison before and after fine-tuning;
   - qualitative rollout grid for the report.

---

## Expected Challenges

- BridgeData V2 actions are not compatible with the original 1X action
  representation.
- Camera streams and frame rates may vary across trajectories.
- Some trajectories may be shorter than 17 frames and should be skipped.
- Domain shift may be stronger than DROID depending on camera viewpoint,
  robot embodiment, and scene layout.

---

## Recommended Claim In The Report

Until BridgeData V2 is fully processed and fine-tuned, describe it as:

> a planned second novel dataset using the same raw-video-to-HWM-token
> preprocessing pipeline validated on DROID.

Do not claim BridgeData V2 quantitative results unless the full preprocessing,
verification, inference, and fine-tuning pipeline has been executed.
