# GFI Domain Context

## Glossary

- **Run configuration**: The resolved settings for one train, inference, video, or analysis run. It includes model selection, dataset presets, paths, flow approximation, init-flow downscale behavior, metrics, and output locations.
- **Interpolation batch**: One batch of frame interpolation work after the dataset loader has produced tensors and before metrics or artifacts are recorded.
- **Flow approximation**: The method that converts source 30fps motion into the init flow used by residual interpolation models.
- **Init flow**: The motion field passed into a residual model before decoder refinement.
- **Dataset preset**: A named collection of ARPG records, modes, difficulties, fps values, and frame-index CSV locations.
- **Model variant**: A selectable interpolation model name such as `IFRNet`, `IFRNet_Residual`, or `IFRNet_Residual_FlowApprox`.
- **Artifact export**: Writing prediction images, flow visualizations, region maps, sample images, or videos from interpolation results.

## Current Architecture Notes

- `scripts/train.py` and `scripts/inference.py` remain formal entrypoints.
- Existing `configs/run/*.yaml` keys and CLI behavior are compatibility requirements.
- Refactors should preserve model behavior and output schemas unless a later design explicitly changes them.
