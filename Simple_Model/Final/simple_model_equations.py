from __future__ import annotations

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm


import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


MOLAR = 1.0
NANOMOLAR = 1e-9 * MOLAR
MICROMOLAR = 1e-6 * MOLAR

class SenderReceiverParams:
    node_length_um: float = 50.0
    center_distance_um: float = 300.0
    bath_margin_um: float = 250.0
    dx_um: float = 10.0
    total_hours: float = 8.0
    dt_s: float = 60.0
    nonlinear_tolerance: float = 1e-9
    max_sweeps_per_step: int = 20

    d_gel_um2_s: float = 60.0
    d_solution_um2_s: float = 150.0
    k_p_s_inv: float = 0.2
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")
        

def apply_preset(params: SenderReceiverParams, preset: str | None) -> SenderReceiverParams:
    params.node_length_um = 75.0
    params.center_distance_um = 175.0
    params.bath_margin_um = 2375.0
    params.d_gel_um2_s = 42.0
    params.d_solution_um2_s = 150.0
    params.k_p_s_inv = 0.2
    params.k_d_ds_s_inv = 3e-4
    params.k_d_ss_s_inv = 3e-4
    params.k_slow_M_inv_s_inv = 1e5
    params.k_fast_M_inv_s_inv = 1e6
    params.sender_switch_nM = 100.0
    params.receiver_switch_nM = 100.0
    params.threshold_uM = 10.0
    return params


# eqs2 = (TransientTerm(var=s2) == k_p - ImplicitSourceTerm(coeff=k_slow * i2, var=s2) - ImplicitSourceTerm(coeff=k_fast * th, var=s2))

# eq_i2 = (TransientTerm(var=i2) == kd_ds * s2_i2 - ImplicitSourceTerm(coeff=k_slow * s2, var=i2))

# eq_th2 = (TransientTerm(var=th) == kd_ds * s2_th - ImplicitSourceTerm(coeff=k_fast * s2, var=th))

# eq_s2_i2 = (TransientTerm(var=s2_i2) ==  k_slow * i2 * s2 - ImplicitSourceTerm(coeff=kd_ds, var=s2_i2))

# eq_s2_th2 = (TransientTerm(var=s2_th) == k_fast * th * s2 - ImplicitSourceTerm(coeff=kd_ds, var=s2_th))

def build_equations(vars_by_name, params: SenderReceiverParams):
    s2 = vars_by_name["S2"]
    i2 = vars_by_name["I2"]
    s2_i2 = vars_by_name["S2_I2"]
    th2 = vars_by_name["Th2"]
    s2_th2 = vars_by_name["S2_Th2"]
    i1o2 = vars_by_name["I1O2"]
    diffusion = vars_by_name["D"]

    eq_s2 = (
        TransientTerm(var=s2)
        == params.k_p_s_inv * i1o2- ImplicitSourceTerm(coeff=(params.k_slow_M_inv_s_inv * i2 + params.k_fast_M_inv_s_inv * th2+ params.k_d_ss_s_inv),var=s2,))

    eq_i2 = (
        TransientTerm(var=i2)
        == params.k_d_ds_s_inv * s2_i2
        - ImplicitSourceTerm(coeff=params.k_slow_M_inv_s_inv * s2, var=i2)
    )

    eq_th2 = (
        TransientTerm(var=th2)
        == params.k_d_ds_s_inv * s2_th2
        - ImplicitSourceTerm(coeff=params.k_fast_M_inv_s_inv * s2, var=th2)
    )

    eq_s2_i2 = (
        TransientTerm(var=s2_i2)
        == params.k_slow_M_inv_s_inv * i2 * s2
        - ImplicitSourceTerm(coeff=params.k_d_ds_s_inv, var=s2_i2)
    )

    eq_s2_th2 = (
        TransientTerm(var=s2_th2)
        == params.k_fast_M_inv_s_inv * th2 * s2
        - ImplicitSourceTerm(coeff=params.k_d_ds_s_inv, var=s2_th2)
    )

    return {
        "S2": eq_s2,
        "I2": eq_i2,
        "Th2": eq_th2,
        "S2_I2": eq_s2_i2,
        "S2_Th2": eq_s2_th2,
    }

def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = True):
    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_h = np.zeros(n_steps + 1)
    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)
    

