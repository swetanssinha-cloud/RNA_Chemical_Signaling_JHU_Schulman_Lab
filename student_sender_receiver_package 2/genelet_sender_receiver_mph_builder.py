#!/usr/bin/env python3
"""
Build a fresh COMSOL sender/receiver model via Python `mph`.

This script creates a simple 2D coefficient-form PDE model in COMSOL that
matches the same sender/receiver equations used in the NumPy and FiPy scripts.
It does not depend on the original paper `.mph` file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

COMSOL_BIN = Path("/Applications/COMSOL64/Multiphysics/bin/macarm64")
if COMSOL_BIN.is_dir():
    os.environ["PATH"] = f"{COMSOL_BIN}:{os.environ.get('PATH', '')}"

import mph  # noqa: E402


def set_parameter(model, name: str, value: str, description: str | None = None):
    model.parameter(name, value)
    if description:
        model.description(name, description)


def apply_preset_to_args(args):
    if not getattr(args, "preset", None):
        return args
    if args.preset == "comsol-2-1":
        args.node_length_um = 75.0
        args.distance_um = 175.0
        args.bath_margin_um = 2375.0
        args.sender_switch_nM = 100.0
        args.receiver_switch_nM = 100.0
        args.threshold_uM = 10.0
        return args
    raise ValueError(f"Unknown preset: {args.preset}")


def coeff_pde_setup(physics, variable_name: str):
    physics.field("dimensionless").field(variable_name)
    physics.field("dimensionless").component([variable_name])


def build_model(model, args):
    java = model.java

    java.component().create("comp1", True)
    java.component("comp1").geom().create("geom1", 2)
    java.component("comp1").mesh().create("mesh1")
    java.component("comp1").geom("geom1").lengthUnit("um")

    # Keep geometry parameters unit-aware so x/y comparisons are correct inside
    # COMSOL. Concentration and kinetic parameters below remain numeric in the
    # same units as the Python scripts.
    set_parameter(model, "L", f"{args.node_length_um}[um]", "Hydrogel node side length")
    set_parameter(model, "xdist", f"{args.distance_um}[um]", "Sender/receiver center distance")
    set_parameter(model, "margin", f"{args.bath_margin_um}[um]", "Bath margin")
    set_parameter(model, "Wbath", "2*margin + xdist + L", "Bath width")
    set_parameter(model, "Hbath", "2*margin + L", "Bath height")
    set_parameter(model, "cy", "Hbath/2", "Common node y center")
    set_parameter(model, "sender_cx", "margin + L/2", "Sender center x")
    set_parameter(model, "receiver_cx", "sender_cx + xdist", "Receiver center x")
    set_parameter(model, "Dgel", "60e-12")
    set_parameter(model, "Dsolution", "150e-12")
    set_parameter(model, "kp", "0.2")
    set_parameter(model, "kd_ds", "3e-4")
    set_parameter(model, "kd_ss", "3e-4")
    set_parameter(model, "kslow", "1e5")
    set_parameter(model, "kfast", "1e6")
    set_parameter(model, "I1O2_0", f"{args.sender_switch_nM * 1e-9:.16g}")
    set_parameter(model, "I2_0", f"{args.receiver_switch_nM * 1e-9:.16g}")
    set_parameter(model, "Th2_0", f"{args.threshold_uM * 1e-6:.16g}")
    set_parameter(model, "t_end", f"{args.hours * 3600.0:.16g}")
    set_parameter(model, "t_step", f"{args.output_dt_s:.16g}")

    geom = java.component("comp1").geom("geom1")
    geom.create("bath", "Rectangle")
    geom.feature("bath").set("size", ["Wbath", "Hbath"])
    geom.feature("bath").set("base", "corner")
    geom.feature("bath").set("pos", ["0", "0"])
    geom.create("sender", "Rectangle")
    geom.feature("sender").set("size", ["L", "L"])
    geom.feature("sender").set("base", "center")
    geom.feature("sender").set("pos", ["sender_cx", "cy"])
    geom.create("receiver", "Rectangle")
    geom.feature("receiver").set("size", ["L", "L"])
    geom.feature("receiver").set("base", "center")
    geom.feature("receiver").set("pos", ["receiver_cx", "cy"])
    geom.create("co1", "Compose")
    geom.feature("co1").selection("input").set(["bath", "sender", "receiver"])
    geom.feature("co1").set("formula", "bath+sender+receiver")
    geom.run()

    java.component("comp1").variable().create("var1")
    var1 = java.component("comp1").variable("var1")
    var1.set(
        "sender_mask",
        "if(abs(x-sender_cx)<=L/2,1,0)*if(abs(y-cy)<=L/2,1,0)",
    )
    var1.set(
        "receiver_mask",
        "if(abs(x-receiver_cx)<=L/2,1,0)*if(abs(y-cy)<=L/2,1,0)",
    )
    var1.set("node_mask", "if(sender_mask+receiver_mask>0,1,0)")
    var1.set("D_s2", "if(node_mask>0,Dgel,Dsolution)")
    var1.set("i2_eff", "if(receiver_mask>0.5,i2,0)")
    var1.set("th2_eff", "if(receiver_mask>0.5,th2,0)")

    comp = java.component("comp1")

    comp.selection().create("sel_sender", "Box")
    comp.selection("sel_sender").set("entitydim", "2")
    comp.selection("sel_sender").set("xmin", "sender_cx - L/4")
    comp.selection("sel_sender").set("xmax", "sender_cx + L/4")
    comp.selection("sel_sender").set("ymin", "cy - L/4")
    comp.selection("sel_sender").set("ymax", "cy + L/4")

    comp.selection().create("sel_receiver", "Box")
    comp.selection("sel_receiver").set("entitydim", "2")
    comp.selection("sel_receiver").set("xmin", "receiver_cx - L/4")
    comp.selection("sel_receiver").set("xmax", "receiver_cx + L/4")
    comp.selection("sel_receiver").set("ymin", "cy - L/4")
    comp.selection("sel_receiver").set("ymax", "cy + L/4")

    comp.physics().create("pdes2", "CoefficientFormPDE", "geom1")
    coeff_pde_setup(comp.physics("pdes2"), "s2")
    comp.physics("pdes2").feature("cfeq1").set("ea", "0")
    comp.physics("pdes2").feature("cfeq1").set("da", "1")
    comp.physics("pdes2").feature("cfeq1").set("c", "D_s2")
    comp.physics("pdes2").feature("cfeq1").set("a", "0")
    comp.physics("pdes2").feature("cfeq1").set(
        "f",
        "kp*I1O2_0*sender_mask - kslow*i2_eff*s2 - kfast*th2_eff*s2 - kd_ss*s2",
    )
    comp.physics("pdes2").feature("init1").set("s2", "0")

    comp.physics().create("dode_i2", "DomainODE", "geom1")
    comp.physics("dode_i2").selection().named("sel_receiver")
    comp.physics("dode_i2").field("dimensionless").field("i2")
    comp.physics("dode_i2").field("dimensionless").component(["i2"])
    comp.physics("dode_i2").feature("dode1").set("f", "kd_ds*s2i2 - kslow*i2*s2")
    comp.physics("dode_i2").feature("init1").set("i2", "I2_0")

    comp.physics().create("dode_s2i2", "DomainODE", "geom1")
    comp.physics("dode_s2i2").selection().named("sel_receiver")
    comp.physics("dode_s2i2").field("dimensionless").field("s2i2")
    comp.physics("dode_s2i2").field("dimensionless").component(["s2i2"])
    comp.physics("dode_s2i2").feature("dode1").set("f", "-kd_ds*s2i2 + kslow*i2*s2")
    comp.physics("dode_s2i2").feature("init1").set("s2i2", "0")

    comp.physics().create("dode_th2", "DomainODE", "geom1")
    comp.physics("dode_th2").selection().named("sel_receiver")
    comp.physics("dode_th2").field("dimensionless").field("th2")
    comp.physics("dode_th2").field("dimensionless").component(["th2"])
    comp.physics("dode_th2").feature("dode1").set("f", "kd_ds*s2th2 - kfast*th2*s2")
    comp.physics("dode_th2").feature("init1").set("th2", "Th2_0")

    comp.physics().create("dode_s2th2", "DomainODE", "geom1")
    comp.physics("dode_s2th2").selection().named("sel_receiver")
    comp.physics("dode_s2th2").field("dimensionless").field("s2th2")
    comp.physics("dode_s2th2").field("dimensionless").component(["s2th2"])
    comp.physics("dode_s2th2").feature("dode1").set("f", "-kd_ds*s2th2 + kfast*th2*s2")
    comp.physics("dode_s2th2").feature("init1").set("s2th2", "0")

    comp.cpl().create("intop_recv", "Integration")
    comp.cpl("intop_recv").selection().named("sel_receiver")

    mesh = java.component("comp1").mesh("mesh1")
    mesh.create("ftri1", "FreeTri")
    mesh.feature("ftri1").create("size1", "Size")
    size = mesh.feature("ftri1").feature("size1")
    if args.hmax_um is not None:
        hmax_m = args.hmax_um * 1e-6
        hmin_m = max(hmax_m / 10.0, 1e-9)
        size.set("custom", "on")
        size.set("hmaxactive", True)
        size.set("hminactive", True)
        size.set("hgradactive", True)
        size.set("hcurveactive", True)
        size.set("hnarrowactive", True)
        size.set("hmax", hmax_m)
        size.set("hmin", hmin_m)
        size.set("hgrad", 1.3)
        size.set("hcurve", 0.3)
        size.set("hnarrow", 1.0)
    else:
        mesh.autoMeshSize(args.mesh_level)
    mesh.run()

    java.study().create("std1")
    java.study("std1").create("time", "Transient")
    java.study("std1").feature("time").activate("pdes2", True)
    java.study("std1").feature("time").activate("dode_i2", True)
    java.study("std1").feature("time").activate("dode_s2i2", True)
    java.study("std1").feature("time").activate("dode_th2", True)
    java.study("std1").feature("time").activate("dode_s2th2", True)
    java.study("std1").feature("time").set("tlist", "range(0,t_step,t_end)")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["comsol-2-1"], default=None)
    parser.add_argument("--distance-um", type=float, default=150.0)
    parser.add_argument("--node-length-um", type=float, default=50.0)
    parser.add_argument("--bath-margin-um", type=float, default=250.0)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--output-dt-s", type=float, default=60.0)
    parser.add_argument("--sender-switch-nM", type=float, default=100.0)
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0)
    parser.add_argument("--threshold-uM", type=float, default=5.0)
    parser.add_argument("--mesh-level", type=int, default=5, help="COMSOL auto mesh size level.")
    parser.add_argument("--hmax-um", type=float, default=None, help="Optional explicit maximum mesh size in um.")
    parser.add_argument(
        "--output-mph",
        type=Path,
        default=Path("reaction_diffusion_models/sender_receiver_built_from_python.mph"),
    )
    parser.add_argument("--no-solve", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args = apply_preset_to_args(args)
    args.output_mph.parent.mkdir(parents=True, exist_ok=True)

    client = mph.start(cores=1)
    model = client.create("sender_receiver_python_built")
    try:
        build_model(model, args)
        if not args.no_solve:
            model.solve()

            try:
                receiver_i2 = model.evaluate("intop_recv(i2)/intop_recv(1)", inner="last")
                receiver_total = model.evaluate(
                    "intop_recv(s2+s2i2+s2th2)/intop_recv(1)",
                    inner="last",
                )
                print(f"receiver_I2_nM {float(receiver_i2) / 1e-9:.6f}")
                print(f"receiver_total_RNA_nM {float(receiver_total) / 1e-9:.6f}")
            except Exception as exc:
                print(f"solve succeeded, but evaluation failed: {exc}")

        model.save(args.output_mph)
        print(f"saved_model {args.output_mph}")
    finally:
        try:
            client.remove(model)
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
