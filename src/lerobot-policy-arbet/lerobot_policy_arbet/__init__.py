"""lerobot_policy_arbet package initialization."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError("lerobot is not installed. Please install lerobot to use lerobot_policy_arbet.") from None

from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from lerobot_policy_arbet.modeling_arbet import ARBeTPolicy
from lerobot_policy_arbet.processor_arbet import make_arbet_pre_post_processors

__all__ = [
    "ARBeTConfig",
    "ARBeTPolicy",
    "make_arbet_pre_post_processors",
]
