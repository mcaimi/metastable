from diffusers.schedulers.scheduling_utils import SchedulerMixin

from metastable.schedulers import FLUX_SCHEDULERS, SCHEDULERS


def test_scheduler_names():
    assert list(SCHEDULERS) == ["DPM++ 2M", "DPM++ SDE", "DPM2", "Euler a", "Euler", "Heun", "LMS"]
    assert list(FLUX_SCHEDULERS) == ["FlowMatch Euler", "FlowMatch Heun"]


def test_registries_are_disjoint():
    assert not set(SCHEDULERS) & set(FLUX_SCHEDULERS)


def test_all_entries_are_instantiable_schedulers():
    for cls in [*SCHEDULERS.values(), *FLUX_SCHEDULERS.values()]:
        assert issubclass(cls, SchedulerMixin)
        cls()  # default config must be valid
