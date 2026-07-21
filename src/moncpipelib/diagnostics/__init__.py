"""Pod resource diagnostics for Kubernetes workloads.

Provides background sampling of CPU and memory metrics from cgroup v2
(or /proc fallback for local development).
"""

from moncpipelib.diagnostics.sampler import PodResourceSampler
from moncpipelib.diagnostics.types import (
    ResourceSample,
    SamplerConfig,
    SamplerMode,
    SamplerSummary,
)

__all__ = [
    "PodResourceSampler",
    "ResourceSample",
    "SamplerConfig",
    "SamplerMode",
    "SamplerSummary",
]
