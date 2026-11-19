#!/usr/bin/env python
"""
Noise Scheduler Registries

Maps human-readable scheduler names to diffusers scheduler classes.
"""

from typing import Dict, Type

from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)


# Discrete-noise schedulers (SD1.5 / SDXL / Illustrious XL)
SCHEDULERS: Dict[str, Type] = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ SDE": DPMSolverSinglestepScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
}


# Flow-matching schedulers for Flux (incompatible with the discrete-noise
# schedulers above). FlowMatchHeunDiscreteScheduler is not offered: the Flux
# pipelines pass custom sigmas to set_timesteps(), which it does not accept.
FLOW_MATCH_SCHEDULERS: Dict[str, Type] = {
    "FlowMatch Euler": FlowMatchEulerDiscreteScheduler,
}
