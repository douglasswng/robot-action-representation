from typing import Any

import torch
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from lerobot_policy_arbet.configuration_arbet import ARBeTConfig

# Keys produced by ScribeTokenDataset that must survive the preprocessor round-trip.
_SCRIBE_KEYS = {"action_tokens", "action_tokens_mask", "action_start"}


def _arbet_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Like the default batch_to_transition, but preserves scribe-specific keys via complementary_data."""
    transition: dict[str, Any] = dict(batch_to_transition(batch))
    comp = transition.get("complementary_data") or {}
    for key in _SCRIBE_KEYS:
        if key in batch:
            comp[key] = batch[key]
    transition["complementary_data"] = comp
    return EnvTransition(**transition)


def _arbet_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """Like the default transition_to_batch — complementary_data (including scribe keys) is already merged."""
    return transition_to_batch(transition)


def make_arbet_pre_post_processors(
    config: ARBeTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for an ARBeT policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Adding a batch dimension.
    3. Moving the data to the specified device.
    4. Normalizing the input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving the data to the CPU.

    Args:
        config: The configuration object for the ARBeT policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_arbet_batch_to_transition,
            to_output=_arbet_transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
