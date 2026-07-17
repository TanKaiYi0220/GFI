from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.flow_approx import build_flow_init_result_with_fill_strategy
from src.engine.flow_approx import flatten_target_index
from src.engine.flow_approx import is_splatting_flow_approx_method
from src.engine.flow_approx import make_source_grid
from src.engine.model_registry import BASELINE_MODEL_NAME
from src.engine.model_registry import uses_flow_approx_model
from src.engine.model_registry import uses_image_only_vfi_model
from src.engine.run_config import FlowApproxConfig
from src.engine.run_config import InferenceRunConfig
from src.engine.run_config import TrainRunConfig

SplattingRegionMaps = dict[str, Any]


@dataclass(frozen=True)
class InterpolationBatchResult:
    img0: Any
    img1: Any
    imgt: Any
    imgt_pred: Any
    embt: Any
    info: dict[str, Any] | None
    bmv: Any | None
    fmv: Any | None
    init_bmv: Any | None
    init_fmv: Any | None
    init_masks: Any | None
    up_flow0_1: Any | None
    up_flow1_1: Any | None
    up_mask_1: Any | None
    imgt_merge: Any | None
    loss_rec: Any | None
    loss_geo: Any | None
    loss_dis: Any | None
    splatting_region_maps: SplattingRegionMaps | None


@dataclass(frozen=True)
class _InitFlowState:
    init_bmv: Any
    init_fmv: Any
    init_masks: Any | None
    init_bmv_mask: Any | None
    init_fmv_mask: Any | None


@dataclass(frozen=True)
class _PreparedBatchInputs:
    img0: Any
    img1: Any
    imgt: Any
    embt: Any
    info: dict[str, Any] | None
    bmv: Any
    fmv: Any
    source_bmv: Any
    source_fmv: Any
    source_depth0: Any | None
    source_depth1: Any | None


@dataclass(frozen=True)
class BenchmarkPhaseBatchInputs:
    inputs: _PreparedBatchInputs
    init_flow: _InitFlowState | None


def _build_exact_imgt_merge(
    img0: Any,
    img1: Any,
    up_flow0_1: Any,
    up_flow1_1: Any,
    up_mask_1: Any,
) -> Any:
    import torch

    from src.models.external.IFRNet.utils import warp

    mean = torch.cat([img0, img1], dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
    img0_centered = img0 - mean
    img1_centered = img1 - mean
    img0_warp = warp(img0_centered, up_flow0_1)
    img1_warp = warp(img1_centered, up_flow1_1)
    return up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean


def _resolve_source_depth_tensors(
    info: dict[str, Any],
    device: Any,
    flow_approx_method: str,
) -> tuple[Any | None, Any | None]:
    if not is_splatting_flow_approx_method(flow_approx_method=flow_approx_method):
        return None, None

    return info["source_depth0"].to(device), info["source_depth1"].to(device)


def _build_init_flow(
    model_name: str,
    source_bmv: Any,
    source_fmv: Any,
    embt: Any,
    flow_approx: FlowApproxConfig,
    source_depth0: Any | None,
    source_depth1: Any | None,
    ground_truth_bmv: Any | None,
    ground_truth_fmv: Any | None,
) -> _InitFlowState:
    if not uses_flow_approx_model(model_name):
        return _InitFlowState(
            init_bmv=source_bmv,
            init_fmv=source_fmv,
            init_masks=None,
            init_bmv_mask=None,
            init_fmv_mask=None,
        )

    flow_init = build_flow_init_result_with_fill_strategy(
        fmv_30=source_fmv,
        bmv_30=source_bmv,
        embt=embt,
        flow_approx_method=flow_approx.method,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        splatting_fill_strategy=flow_approx.splatting_fill_strategy,
        ground_truth_bmv=ground_truth_bmv,
        ground_truth_fmv=ground_truth_fmv,
    )
    init_bmv_mask = None
    init_fmv_mask = None
    if flow_approx.init_flow_downscale_strategy == "masked_area":
        if flow_init.masks is None:
            raise RuntimeError(
                "init_flow_downscale_strategy=masked_area requires splatting coverage masks, but none were produced."
            )
        init_bmv_mask = flow_init.masks[:, 0:1]
        init_fmv_mask = flow_init.masks[:, 1:2]

    return _InitFlowState(
        init_bmv=flow_init.bmv,
        init_fmv=flow_init.fmv,
        init_masks=flow_init.masks,
        init_bmv_mask=init_bmv_mask,
        init_fmv_mask=init_fmv_mask,
    )


def _run_model_forward(
    model_name: str,
    model: Any,
    img0: Any,
    img1: Any,
    embt: Any,
    imgt: Any,
    source_bmv: Any,
    source_fmv: Any,
    flow_approx: FlowApproxConfig,
    source_depth0: Any | None,
    source_depth1: Any | None,
    ground_truth_bmv: Any | None,
    ground_truth_fmv: Any | None,
) -> tuple[Any, _InitFlowState]:
    import torch

    if uses_image_only_vfi_model(model_name):
        init_flow = _InitFlowState(
            init_bmv=None,
            init_fmv=None,
            init_masks=None,
            init_bmv_mask=None,
            init_fmv_mask=None,
        )
        return model(img0, img1, embt, imgt), init_flow

    init_flow = _build_init_flow(
        model_name=model_name,
        source_bmv=source_bmv,
        source_fmv=source_fmv,
        embt=embt,
        flow_approx=flow_approx,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
        ground_truth_bmv=ground_truth_bmv,
        ground_truth_fmv=ground_truth_fmv,
    )

    if model_name == BASELINE_MODEL_NAME:
        flow = torch.cat([init_flow.init_bmv, init_flow.init_fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow), init_flow

    return (
        model(
            img0,
            img1,
            embt,
            imgt,
            init_flow0=init_flow.init_bmv,
            init_flow1=init_flow.init_fmv,
            init_flow0_mask=init_flow.init_bmv_mask,
            init_flow1_mask=init_flow.init_fmv_mask,
            init_flow_mask_epsilon=flow_approx.init_flow_mask_epsilon,
        ),
        init_flow,
    )


def build_nearest_splat_hit_count(source_motion: Any) -> Any:
    import torch

    batch_size = int(source_motion.shape[0])
    height = int(source_motion.shape[2])
    width = int(source_motion.shape[3])
    pixel_count = height * width
    source_grid = make_source_grid(batch_size, height, width, source_motion.device, source_motion.dtype)
    target_position = source_grid + source_motion
    target_x = target_position[:, 0].round().long()
    target_y = target_position[:, 1].round().long()
    valid = (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
    flat_target_index = flatten_target_index(
        target_x=target_x.clamp(0, width - 1),
        target_y=target_y.clamp(0, height - 1),
        width=width,
    ).reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1).to(dtype=source_motion.dtype)
    hit_count = torch.zeros(
        (batch_size, pixel_count),
        device=source_motion.device,
        dtype=source_motion.dtype,
    )
    hit_count.scatter_add_(dim=1, index=flat_target_index, src=flat_valid)
    return hit_count.reshape(batch_size, 1, height, width)


def build_splatting_region_maps(
    fmv_30: Any,
    bmv_30: Any,
    embt: Any,
    init_masks: Any | None,
) -> SplattingRegionMaps | None:
    if init_masks is None:
        return None

    time = embt.reshape(embt.shape[0], 1, 1, 1)
    bmv_hit_count = build_nearest_splat_hit_count(time * fmv_30)
    fmv_hit_count = build_nearest_splat_hit_count((1 - time) * bmv_30)
    bmv_hit = init_masks[:, 0:1] > 0
    fmv_hit = init_masks[:, 1:2] > 0
    hit_both = bmv_hit & fmv_hit
    hole_any = ~hit_both
    bmv_many_to_one = bmv_hit_count > 1
    fmv_many_to_one = fmv_hit_count > 1
    many_to_one_any = bmv_many_to_one | fmv_many_to_one
    return {
        "bmv_hit": bmv_hit,
        "fmv_hit": fmv_hit,
        "hit_both": hit_both,
        "hole_any": hole_any,
        "bmv_many_to_one": bmv_many_to_one,
        "fmv_many_to_one": fmv_many_to_one,
        "many_to_one_any": many_to_one_any,
        "bmv_hit_count": bmv_hit_count,
        "fmv_hit_count": fmv_hit_count,
    }


def _prepare_batch_inputs(
    model_name: str,
    batch: Any,
    device: Any,
    flow_approx_method: str,
) -> _PreparedBatchInputs:
    try:
        if uses_flow_approx_model(model_name):
            img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, embt, info = batch
            source_bmv = bmv_30.to(device)
            source_fmv = fmv_30.to(device)
            bmv = bmv_60.to(device)
            fmv = fmv_60.to(device)
            source_depth0, source_depth1 = _resolve_source_depth_tensors(
                info=info,
                device=device,
                flow_approx_method=flow_approx_method,
            )
        else:
            img0, imgt, img1, bmv, fmv, embt, info = batch
            source_bmv = bmv.to(device)
            source_fmv = fmv.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            source_depth0 = None
            source_depth1 = None
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported batch contract for model_name={model_name}") from error

    return _PreparedBatchInputs(
        img0=img0.to(device),
        img1=img1.to(device),
        imgt=imgt.to(device),
        embt=embt.to(device),
        info=info,
        bmv=bmv,
        fmv=fmv,
        source_bmv=source_bmv,
        source_fmv=source_fmv,
        source_depth0=source_depth0,
        source_depth1=source_depth1,
    )


def _run_model_inference(
    model_name: str,
    model: Any,
    batch_inputs: _PreparedBatchInputs,
    flow_approx: FlowApproxConfig,
    scale_factor: float,
) -> InterpolationBatchResult:
    if uses_image_only_vfi_model(model_name):
        imgt_pred = model.inference(
            batch_inputs.img0,
            batch_inputs.img1,
            batch_inputs.embt,
            scale_factor,
        )
        return InterpolationBatchResult(
            img0=batch_inputs.img0,
            img1=batch_inputs.img1,
            imgt=batch_inputs.imgt,
            imgt_pred=imgt_pred,
            embt=batch_inputs.embt,
            info=batch_inputs.info,
            bmv=None,
            fmv=None,
            init_bmv=None,
            init_fmv=None,
            init_masks=None,
            up_flow0_1=None,
            up_flow1_1=None,
            up_mask_1=None,
            imgt_merge=None,
            loss_rec=None,
            loss_geo=None,
            loss_dis=None,
            splatting_region_maps=None,
        )

    if model_name == BASELINE_MODEL_NAME:
        imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(
            batch_inputs.img0,
            batch_inputs.img1,
            batch_inputs.embt,
            scale_factor,
        )
        return InterpolationBatchResult(
            img0=batch_inputs.img0,
            img1=batch_inputs.img1,
            imgt=batch_inputs.imgt,
            imgt_pred=imgt_pred,
            embt=batch_inputs.embt,
            info=batch_inputs.info,
            bmv=batch_inputs.bmv,
            fmv=batch_inputs.fmv,
            init_bmv=None,
            init_fmv=None,
            init_masks=None,
            up_flow0_1=up_flow0_1,
            up_flow1_1=up_flow1_1,
            up_mask_1=up_mask_1,
            imgt_merge=None,
            loss_rec=None,
            loss_geo=None,
            loss_dis=None,
            splatting_region_maps=None,
        )

    init_flow = _build_init_flow(
        model_name=model_name,
        source_bmv=batch_inputs.source_bmv,
        source_fmv=batch_inputs.source_fmv,
        embt=batch_inputs.embt,
        flow_approx=flow_approx,
        source_depth0=batch_inputs.source_depth0,
        source_depth1=batch_inputs.source_depth1,
        ground_truth_bmv=batch_inputs.bmv,
        ground_truth_fmv=batch_inputs.fmv,
    )
    return _run_model_inference_with_init_flow(
        model_name=model_name,
        model=model,
        batch_inputs=batch_inputs,
        flow_approx=flow_approx,
        scale_factor=scale_factor,
        init_flow=init_flow,
    )


def _run_model_inference_with_init_flow(
    model_name: str,
    model: Any,
    batch_inputs: _PreparedBatchInputs,
    flow_approx: FlowApproxConfig,
    scale_factor: float,
    init_flow: _InitFlowState | None,
) -> InterpolationBatchResult:
    if init_flow is None:
        raise ValueError(f"Benchmark/model inference requires prebuilt init_flow for model_name={model_name}")

    imgt_pred, up_flow0_1, up_flow1_1, up_mask_1, _up_res_1, imgt_merge = model.inference(
        batch_inputs.img0,
        batch_inputs.img1,
        batch_inputs.embt,
        scale_factor,
        init_flow0=init_flow.init_bmv,
        init_flow1=init_flow.init_fmv,
        init_flow0_mask=init_flow.init_bmv_mask,
        init_flow1_mask=init_flow.init_fmv_mask,
        init_flow_mask_epsilon=flow_approx.init_flow_mask_epsilon,
    )
    return InterpolationBatchResult(
        img0=batch_inputs.img0,
        img1=batch_inputs.img1,
        imgt=batch_inputs.imgt,
        imgt_pred=imgt_pred,
        embt=batch_inputs.embt,
        info=batch_inputs.info,
        bmv=batch_inputs.bmv,
        fmv=batch_inputs.fmv,
        init_bmv=init_flow.init_bmv,
        init_fmv=init_flow.init_fmv,
        init_masks=init_flow.init_masks,
        up_flow0_1=up_flow0_1,
        up_flow1_1=up_flow1_1,
        up_mask_1=up_mask_1,
        imgt_merge=imgt_merge,
        loss_rec=None,
        loss_geo=None,
        loss_dis=None,
        splatting_region_maps=build_splatting_region_maps(
            batch_inputs.source_fmv,
            batch_inputs.source_bmv,
            batch_inputs.embt,
            init_flow.init_masks,
        ),
    )


def prepare_benchmark_batch_inputs(config: InferenceRunConfig, batch: Any, device: Any) -> BenchmarkPhaseBatchInputs:
    inputs = _prepare_batch_inputs(
        model_name=config.model.model_name,
        batch=batch,
        device=device,
        flow_approx_method=config.flow_approx.method,
    )
    return BenchmarkPhaseBatchInputs(inputs=inputs, init_flow=None)


def build_benchmark_init_flow(
    config: InferenceRunConfig,
    phase_inputs: BenchmarkPhaseBatchInputs,
) -> BenchmarkPhaseBatchInputs:
    if uses_image_only_vfi_model(config.model.model_name) or not uses_flow_approx_model(config.model.model_name):
        return phase_inputs

    init_flow = _build_init_flow(
        model_name=config.model.model_name,
        source_bmv=phase_inputs.inputs.source_bmv,
        source_fmv=phase_inputs.inputs.source_fmv,
        embt=phase_inputs.inputs.embt,
        flow_approx=config.flow_approx,
        source_depth0=phase_inputs.inputs.source_depth0,
        source_depth1=phase_inputs.inputs.source_depth1,
        ground_truth_bmv=phase_inputs.inputs.bmv,
        ground_truth_fmv=phase_inputs.inputs.fmv,
    )
    return BenchmarkPhaseBatchInputs(inputs=phase_inputs.inputs, init_flow=init_flow)


def run_benchmark_model_phase(
    config: InferenceRunConfig,
    model: Any,
    batch_inputs: BenchmarkPhaseBatchInputs,
    init_flow: Any,
) -> Any:
    if uses_image_only_vfi_model(config.model.model_name):
        return _run_model_inference(
            model_name=config.model.model_name,
            model=model,
            batch_inputs=batch_inputs.inputs,
            flow_approx=config.flow_approx,
            scale_factor=config.scale_factor,
        )

    if config.model.model_name == BASELINE_MODEL_NAME:
        return _run_model_inference(
            model_name=config.model.model_name,
            model=model,
            batch_inputs=batch_inputs.inputs,
            flow_approx=config.flow_approx,
            scale_factor=config.scale_factor,
        )

    resolved_init_flow = init_flow
    if isinstance(init_flow, BenchmarkPhaseBatchInputs):
        resolved_init_flow = init_flow.init_flow
    if resolved_init_flow is None:
        resolved_init_flow = batch_inputs.init_flow
    return _run_model_inference_with_init_flow(
        model_name=config.model.model_name,
        model=model,
        batch_inputs=batch_inputs.inputs,
        flow_approx=config.flow_approx,
        scale_factor=config.scale_factor,
        init_flow=resolved_init_flow,
    )


def run_training_batch(
    config: TrainRunConfig,
    model: Any,
    batch: Any,
    device: Any,
    collect_visual_artifacts: bool,
) -> InterpolationBatchResult:
    model_name = config.model.model_name
    batch_inputs = _prepare_batch_inputs(
        model_name=model_name,
        batch=batch,
        device=device,
        flow_approx_method=config.flow_approx.method,
    )

    model_output, init_flow = _run_model_forward(
        model_name=model_name,
        model=model,
        img0=batch_inputs.img0,
        img1=batch_inputs.img1,
        embt=batch_inputs.embt,
        imgt=batch_inputs.imgt,
        source_bmv=batch_inputs.source_bmv,
        source_fmv=batch_inputs.source_fmv,
        flow_approx=config.flow_approx,
        source_depth0=batch_inputs.source_depth0,
        source_depth1=batch_inputs.source_depth1,
        ground_truth_bmv=batch_inputs.bmv,
        ground_truth_fmv=batch_inputs.fmv,
    )
    imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model_output
    image_only_model = uses_image_only_vfi_model(model_name)
    if model_name == BASELINE_MODEL_NAME or image_only_model:
        init_bmv = None
        init_fmv = None
        init_masks = None
        imgt_merge = None
        splatting_region_maps = None
    else:
        init_bmv = init_flow.init_bmv
        init_fmv = init_flow.init_fmv
        init_masks = init_flow.init_masks
        if collect_visual_artifacts:
            imgt_merge = _build_exact_imgt_merge(
                img0=batch_inputs.img0,
                img1=batch_inputs.img1,
                up_flow0_1=up_flow0_1,
                up_flow1_1=up_flow1_1,
                up_mask_1=up_mask_1,
            )
            splatting_region_maps = build_splatting_region_maps(
                fmv_30=batch_inputs.source_fmv,
                bmv_30=batch_inputs.source_bmv,
                embt=batch_inputs.embt,
                init_masks=init_masks,
            )
        else:
            imgt_merge = None
            splatting_region_maps = None

    return InterpolationBatchResult(
        img0=batch_inputs.img0,
        img1=batch_inputs.img1,
        imgt=batch_inputs.imgt,
        imgt_pred=imgt_pred,
        embt=batch_inputs.embt,
        info=batch_inputs.info,
        bmv=None if image_only_model else batch_inputs.bmv,
        fmv=None if image_only_model else batch_inputs.fmv,
        init_bmv=init_bmv,
        init_fmv=init_fmv,
        init_masks=init_masks,
        up_flow0_1=up_flow0_1,
        up_flow1_1=up_flow1_1,
        up_mask_1=up_mask_1,
        imgt_merge=imgt_merge,
        loss_rec=loss_rec,
        loss_geo=loss_geo,
        loss_dis=loss_dis,
        splatting_region_maps=splatting_region_maps,
    )


def run_training_sample_batch(
    config: TrainRunConfig,
    model: Any,
    batch: Any,
    device: Any,
) -> InterpolationBatchResult:
    batch_inputs = _prepare_batch_inputs(
        model_name=config.model.model_name,
        batch=batch,
        device=device,
        flow_approx_method=config.flow_approx.method,
    )
    return _run_model_inference(
        model_name=config.model.model_name,
        model=model,
        batch_inputs=batch_inputs,
        flow_approx=config.flow_approx,
        scale_factor=1.0,
    )


def run_inference_batch(
    config: InferenceRunConfig,
    model: Any,
    batch: Any,
    device: Any,
) -> InterpolationBatchResult:
    batch_inputs = _prepare_batch_inputs(
        model_name=config.model.model_name,
        batch=batch,
        device=device,
        flow_approx_method=config.flow_approx.method,
    )
    return _run_model_inference(
        model_name=config.model.model_name,
        model=model,
        batch_inputs=batch_inputs,
        flow_approx=config.flow_approx,
        scale_factor=config.scale_factor,
    )
