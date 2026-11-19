from diffusers.schedulers.scheduling_utils import SchedulerMixin

from metastable.schedulers import FLOW_MATCH_SCHEDULERS, SCHEDULERS


def test_scheduler_names():
    assert list(SCHEDULERS) == ["DPM++ 2M", "DPM++ SDE", "DPM2", "Euler a", "Euler", "Heun", "LMS"]
    assert list(FLOW_MATCH_SCHEDULERS) == ["FlowMatch Euler"]


def test_registries_are_disjoint():
    assert not set(SCHEDULERS) & set(FLOW_MATCH_SCHEDULERS)


def test_all_entries_are_instantiable_schedulers():
    for cls in [*SCHEDULERS.values(), *FLOW_MATCH_SCHEDULERS.values()]:
        assert issubclass(cls, SchedulerMixin)
        cls()  # default config must be valid



def test_flow_match_schedulers_accept_pipeline_timesteps():
    """Flux and Z-Image pipelines always pass custom sigmas (and mu) to set_timesteps()."""
    from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps as flux_retrieve
    from diffusers.pipelines.z_image.pipeline_z_image import (
        get_default_z_image_sigmas,
        retrieve_timesteps as zimage_retrieve,
    )

    sigmas = get_default_z_image_sigmas(8)
    for cls in FLOW_MATCH_SCHEDULERS.values():
        for retrieve in (flux_retrieve, zimage_retrieve):
            timesteps, steps = retrieve(cls(), 8, "cpu", sigmas=sigmas, mu=0.8)
            assert steps == 8 and len(timesteps) == 8


def test_flow_match_heun_rejects_custom_sigmas():
    """Why FlowMatchHeunDiscreteScheduler is not offered."""
    import pytest
    from diffusers import FlowMatchHeunDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import retrieve_timesteps

    with pytest.raises(ValueError, match="sigmas"):
        retrieve_timesteps(FlowMatchHeunDiscreteScheduler(), 8, "cpu", sigmas=[1.0, 0.5], mu=0.8)
