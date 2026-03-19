from dataclasses import replace
from typing import cast

import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from lerobot_policy_arbet.processor_arbet import make_arbet_pre_post_processors

_BASE_CONFIG = ARBeTConfig(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    },
)


def _make_config(**overrides) -> ARBeTConfig:
    return replace(_BASE_CONFIG, **overrides)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestMakeProcessors:
    def test_returns_two_pipelines(self):
        pre, post = make_arbet_pre_post_processors(_make_config())
        assert type(pre).__name__ == "DataProcessorPipeline"
        assert type(post).__name__ == "DataProcessorPipeline"

    def test_pipeline_names(self):
        pre, post = make_arbet_pre_post_processors(_make_config())
        assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
        assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME


# ---------------------------------------------------------------------------
# Preprocessor step types
# ---------------------------------------------------------------------------


class TestPreprocessorSteps:
    def test_step_types_and_order(self):
        pre, _ = make_arbet_pre_post_processors(_make_config())
        step_types = [type(s) for s in pre.steps]
        assert step_types == [
            RenameObservationsProcessorStep,
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            NormalizerProcessorStep,
        ]

    def test_device_step_uses_config_device(self):
        cfg = _make_config()
        pre, _ = make_arbet_pre_post_processors(cfg)
        device_step = cast(DeviceProcessorStep, pre.steps[2])
        assert device_step.device == cfg.device

    def test_normalizer_includes_all_features(self):
        cfg = _make_config()
        assert cfg.input_features is not None
        assert cfg.output_features is not None
        pre, _ = make_arbet_pre_post_processors(cfg)
        norm_step = cast(NormalizerProcessorStep, pre.steps[3])
        expected_keys = set(cfg.input_features) | set(cfg.output_features)
        assert set(norm_step.features) == expected_keys


# ---------------------------------------------------------------------------
# Postprocessor step types
# ---------------------------------------------------------------------------


class TestPostprocessorSteps:
    def test_step_types_and_order(self):
        _, post = make_arbet_pre_post_processors(_make_config())
        step_types = [type(s) for s in post.steps]
        assert step_types == [
            UnnormalizerProcessorStep,
            DeviceProcessorStep,
        ]

    def test_unnormalizer_covers_output_features(self):
        cfg = _make_config()
        assert cfg.output_features is not None
        _, post = make_arbet_pre_post_processors(cfg)
        unnorm_step = cast(UnnormalizerProcessorStep, post.steps[0])
        assert set(unnorm_step.features) == set(cfg.output_features)

    def test_device_step_moves_to_cpu(self):
        _, post = make_arbet_pre_post_processors(_make_config())
        device_step = cast(DeviceProcessorStep, post.steps[1])
        assert device_step.device == "cpu"


# ---------------------------------------------------------------------------
# Dataset stats forwarding
# ---------------------------------------------------------------------------


class TestDatasetStats:
    def test_none_stats_accepted(self):
        pre, post = make_arbet_pre_post_processors(_make_config(), dataset_stats=None)
        assert pre is not None
        assert post is not None

    def test_stats_forwarded_to_normalizer(self):
        fake_stats = {
            OBS_STATE: {"mean": torch.zeros(2), "std": torch.ones(2)},
            ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)},
        }
        cfg = _make_config()
        pre, post = make_arbet_pre_post_processors(cfg, dataset_stats=fake_stats)
        norm_step = cast(NormalizerProcessorStep, pre.steps[3])
        assert norm_step.stats is fake_stats
        unnorm_step = cast(UnnormalizerProcessorStep, post.steps[0])
        assert unnorm_step.stats is fake_stats


# ---------------------------------------------------------------------------
# Action normalization is IDENTITY
# ---------------------------------------------------------------------------


class TestActionIdentityNormalization:
    def test_action_norm_mode_is_identity(self):
        cfg = _make_config()
        assert cfg.normalization_mapping["ACTION"] == NormalizationMode.IDENTITY
