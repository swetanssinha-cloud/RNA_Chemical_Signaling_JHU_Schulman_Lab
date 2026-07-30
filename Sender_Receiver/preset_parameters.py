import numpy as np
import matplotlib as plt
from dataclasses import dataclass
import argparse


@dataclass
class SenderReceiverParams:
    node_length_um: float = 50.0 #um
    center_distance_um: float = 1500.0 #um
    bath_margin_um: float = 250.0 #um

    dx_um: float = 10.0 #um
    node_diameter = 75

    total_hours: float = 8 #hrs
    dt_s: float = 60.0 #s

    nonlinear_tolerance: float = 1e-9
    max_sweeps_per_step: int = 20

    d_gel_um2_s: float = 60.0 #(um^2)/s
    d_solution_um2_s: float = 150.0 #(um^2)/s
    k_p_s_inv: float = 0.2 #1/s
    k_d_ds_s_inv: float = 3e-4 #1/s
    k_d_ss_s_inv: float = 3e-4 #1/s
    k_slow_M_inv_s_inv: float = 1e5 #1/(Ms)
    k_fast_M_inv_s_inv: float = 1e6 #1/(Ms)

    sender_switch_nM: float = 100.0 #nM
    receiver_switch_nM: float = 100.0 #nM
    threshold_uM: float = 5.0 #uM

    total_height = 1e3
    total_width = 1e4 

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")


def apply_preset(params: SenderReceiverParams, preset: str | None) -> SenderReceiverParams:
    if preset == 'Chloe_System':
        params.node_diameter = 75
        params.center_distance_um = 200
        params.d_gel_um2_s = 60.0
        params.d_solution_um2_s = 150.0
        params.k_p_s_inv = 0.2
        params.k_d_ds_s_inv = 3e-4
        params.k_d_ss_s_inv = 3e-4
        params.k_slow_M_inv_s_inv = 1e5
        params.k_fast_M_inv_s_inv = 1e6
        params.threshold_uM = 5.0   

        params.total_height = 1e3
        params.total_width = 1e4

    if preset == 'Chen_System':
        params.total_height = 5000
        params.total_width = 5000
        params.node_length_um = 50
        params.center_distance_um = 200
        params.d_gel_um2_s = 60.0
        params.d_solution_um2_s = 150.0
        params.k_p_s_inv = 0.2
        params.k_d_ds_s_inv = 3e-4
        params.k_d_ss_s_inv = 3e-4
        params.k_slow_M_inv_s_inv = 1e5
        params.k_fast_M_inv_s_inv = 1e6
        params.threshold_uM = 5.0 

        