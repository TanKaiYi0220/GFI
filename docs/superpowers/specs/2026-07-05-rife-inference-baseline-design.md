# RIFE Inference Baseline Design

## Goal

Add official RIFE as the first external video-frame-interpolation baseline in the runtime-refactor branch.

This first slice supports inference and evaluation only. It does not add RIFE fine-tuning.

## Source And Local Setup

Use the official PyTorch RIFE repository:

- `https://github.com/hzwer/ECCV2022-RIFE`

The official repo and checkpoint are local dependencies, not committed GFI source:

- Clone target: `src/models/external/RIFE`
- Checkpoint target: `src/models/external/RIFE/train_log/flownet.pkl`
- `src/models/external/` remains gitignored.

Do not add a committed setup helper script. The external repo and checkpoint are prepared manually on the target machine. Verification may create a local ignored clone under `src/models/external/RIFE`, but that clone must not be committed and must not overwrite an existing directory. The implementation should not silently download checkpoints, because official checkpoint links may require Drive or Baidu access.

## Scope

In scope:

- Add `model_name: "RIFE"` for inference/evaluation.
- Add a GFI-side RIFE adapter.
- Run RIFE predictions through the existing inference metrics path.
- Preserve existing inference config style and output conventions where compatible.
- Keep missing external repo/checkpoint errors explicit and actionable.

Out of scope:

- RIFE fine-tuning.
- Committing the official RIFE repo or checkpoint into GFI.
- Adapting UPRNet, EMA-VFI, or other baselines.
- Flow-analysis parity with IFRNet.

## Adapter Contract

Add a committed adapter at `src/models/RIFE.py` that wraps the official RIFE model behind the GFI inference contract.

The adapter should:

- Load official RIFE code from `src/models/external/RIFE`.
- Load `flownet.pkl` through official RIFE loading behavior or a minimal equivalent.
- Expose an inference method compatible with the engine call site:
  - Inputs: `img0`, `img1`, `embt`, `scale_factor`.
  - Output: interpolated frame tensor `imgt_pred`.
- Read the timestep from `embt` when possible. The first slice supports one scalar timestep per batch. If a batch contains mixed per-sample timesteps, fail fast with a clear unsupported-timestep error.
- Avoid exposing training behavior in this slice.

RIFE is treated as an image-interpolation baseline. It is not treated as a flow-analysis model.

## Runtime Integration

Extend the runtime-refactor seams rather than adding script-local RIFE branches.

`src.engine.model_registry`:

- Add `RIFE_MODEL_NAME = "RIFE"`.
- Include RIFE in inference-capable model names.
- Resolve it to `src.models.RIFE.Model`.
- Keep RIFE out of flow-approx model names.

`src.engine.interpolation_batch`:

- Add a path for image-only VFI models.
- Use the existing baseline dataset batch contract: `img0`, `imgt`, `img1`, `bmv`, `fmv`, `embt`, `info`.
- Call RIFE inference and return `InterpolationBatchResult` with:
  - `img0`, `img1`, `imgt`, `imgt_pred`, `embt`, `info` populated.
  - Flow, mask, init-flow, loss, merge, and splatting-region fields set to `None`.
- Keep flow-specific validation at the owner seam. Do not repeat validation through intermediate helpers.

`src.engine.checkpoints`:

- Support RIFE checkpoint loading through the adapter or a model-owned loading hook.
- Preserve existing checkpoint behavior for IFRNet variants.
- Fail fast if the RIFE checkpoint path is not a directory containing `flownet.pkl`, or if official RIFE files are missing.

`scripts/inference.py`:

- Avoid RIFE internals.
- Continue to own metric accumulation and CSV writing.
- Skip or leave blank flow-specific selected artifact fields for RIFE.
- Raise a clear error only when the user explicitly requests a flow-specific artifact or metric that RIFE cannot provide.

## Config

Add a representative RIFE inference config:

- `configs/run/inference_rife_official.yaml`

Expected key choices:

- `model_name: "RIFE"`
- `checkpoint_path: "src/models/external/RIFE/train_log"`
- Same dataset root, preset, FPS, output, and metric structure used by existing inference configs.
- Enable PSNR and SSIM by default for the representative RIFE config.
- Disable LPIPS, VFIPS, FLOLPIPS, PSNR-div, and flow-specific selection by default in the representative RIFE config.
- Flow-specific artifacts and flow-diff selection should be disabled by default.

Dry-run output should remain parseable JSON and include `model_name: "RIFE"`.

## Error Handling

The implementation should fail fast with actionable messages for:

- Missing `src/models/external/RIFE`.
- Missing official model files required by the adapter.
- Missing `train_log/flownet.pkl`.
- Unsupported RIFE timestep shape.
- Flow-specific artifact or metric requests that require fields RIFE does not return.

The implementation should not silently fall back to another model, another checkpoint path, or empty predictions.

## Verification

Required verification:

- Import smoke:
  - `python -c "import src.models.RIFE; import scripts.inference"`
- Optional local clone for verification only:
  - If `src/models/external/RIFE` is absent in the verification environment, clone `https://github.com/hzwer/ECCV2022-RIFE` into that ignored path before functional smoke checks.
  - Do not commit the clone and do not overwrite an existing external directory.
- External dependency failure smoke:
  - Missing external repo/checkpoint produces the expected actionable error.
- Dry-run:
  - `python scripts/inference.py --config configs/run/inference_rife_official.yaml` with dry-run mode, or an equivalent temporary dry-run config, exits `0` and prints parseable JSON.
- Compile:
  - `python -m compileall -q src scripts tests`

If `src/models/external/RIFE/train_log/flownet.pkl` exists:

- Run a tiny inference smoke over one batch or a minimal synthetic batch.
- Verify `imgt_pred` shape matches `imgt`.
- Verify image metrics can run.
- Verify flow-specific artifact fields are skipped or blank instead of crashing.

## Later Fine-Tuning Slice

Fine-tuning should be designed separately after inference/evaluation is stable.

The later slice should compare:

- Dataset split and preprocessing.
- Crop and augmentation policy.
- Optimizer and schedule.
- Training budget.
- RIFE-native losses versus GFI/IFRNet losses.
- Checkpoint save/load format.

Do not force RIFE fine-tuning into the IFRNet training loop in this first slice.
