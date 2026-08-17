"""
Shared model, solver, and sweep orchestration for the on-then-off variant of
the 2-node tethered-genelet sender/receiver model.

This is NOT a second copy of Single_parameter_sweeps/sweep_core.py. Every
per-parameter sweep script in this folder imports from HERE, and this module
in turn imports the original as `base` and reuses it directly wherever the
on/off logic doesn't change anything: DEFAULT_PARAMS, mesh handling
(build_all_meshes, mesh_path_for -- including the shared mesh cache, so e.g.
distance_between doesn't rebuild meshes the other sweep already made),
apply_sweep_value, initialize_fields, build_S2_equation, reaction_pair_step,
half_time, scalars_from_timeseries, summarise, plot, and the validation-gate
philosophy. The only thing rewritten here is run_single_simulation (fixed
8h -> adaptive two-phase on/off), run_sweep (has to call the new
run_single_simulation), run() (orchestrates the new run_sweep), and
plot_timeseries (adds a shutoff marker per curve).

(The module can't literally be named sweep_core.py like the original --
Python would try to import itself when this file does `import sweep_core`.)


WHY ADAPTIVE, NOT A FIXED SHUTOFF TIME
---------------------------------------
On_then_off.py settled on a flat 6h cutoff for the single default-parameter
run it was built for. That number is specific to k_d_ds = k_d_ss = 3e-4 (the
DEFAULT_PARAMS values). Two of these five sweeps vary k_d_ds or k_d_ss
themselves, down to np.linspace(0, 0.1, 50) * 3e-4 -- i.e. 0 up to 3e-5, at
least 10x slower than the value the 6h number was tuned against. A flat 6h
cutoff would cut those runs off mid-decline, not at a floor, which defeats
the point of comparing the off/on behaviour across the sweep. So every run
here reuses On_then_off.py's own logic instead: turn I1O2 off when the
receiver reaches steady state (checked only on cells inside the receiver
node -- the whole-mesh version of this check is what stalled the original
On_then_off.py attempts, see that file's docstring), with a generous
per-phase timeout (on_phase_max_time / off_phase_max_time, default 48h / 150h
-- see SweepConfig for why they differ) as a backstop for parameter
values that never numerically settle. Because it's the same criterion at
every sweep point, "when did it turn off" becomes part of the answer instead
of an assumption baked into the experiment design.


OUTPUT LAYOUT
-------------
Results land in sweep_<parameter>_on_off/ inside THIS folder -- not in
Paramter_sweep/ itself -- so nothing here can collide with or overwrite the
existing (no-shutoff) sweeps in Single_parameter_sweeps/. Meshes are still
shared: base.MESH_DIR resolves to Paramter_sweep/meshes_conformal regardless
of which sweep_core module built it.


NEW METRICS
-----------
None yet, by design -- this only adds the timeseries overlay (with a shutoff
marker) on top of the same summary_stats.csv / raw_results.csv / plot() that
the no-shutoff sweeps already produce (still meaningful here: "final" now
means "after the recovery phase", and half_time_center_hr spans the full
on+off run). Recovery-specific metrics (floor depth, recovered level,
recovery half-time) can be added once the actual shapes across a sweep are
visible.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fipy import Gmsh2D, LinearLUSolver

THIS_DIR = Path(__file__).resolve().parent                 # .../sweep_one_parameter_turn_on_off
PARAM_SWEEP_ROOT = THIS_DIR.parent                          # .../Paramter_sweep
OUTPUT_ROOT = THIS_DIR

sys.path.insert(0, str(PARAM_SWEEP_ROOT / "Single_parameter_sweeps"))
import sweep_core as base   # noqa: E402  -- the original, unmodified module


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

@dataclass
class SweepConfig:
    """
    Same shape as base.SweepConfig, plus the adaptive on/off phase controls.

    output_dir defaults to sweep_<parameter>_on_off INSIDE this folder (not
    Paramter_sweep/ -- see module docstring). n_processes defaults the same
    way base's does (SLURM_CPUS_PER_TASK if set, else cpu_count(), capped at
    the number of tasks).
    """
    sweep_parameter: str
    sweep_values: Sequence[float]
    n_replicates: int = 1
    output_dir: Optional[Path] = None
    n_processes: Optional[int] = None

    # Adaptive phase control -- same criterion tuned in On_then_off.py
    # (receiver-local relative-change check), reused unchanged at every
    # sweep point. ON and OFF have separate caps because they are not
    # symmetric in practice: across the first k_d_ds sweep, on_converged was
    # True for all 50 runs (observed range 20-39h, comfortably under 48h),
    # but off_converged was True for only 1 of 50 -- the other 49 all
    # stopped at exactly t_shutoff + 48h, i.e. the timeout was firing, not
    # genuine convergence. Recovery is just slower than the ON-phase decline
    # (see On_then_off.py's docstring on why), so it gets a much bigger
    # budget.
    check_interval: int = 50          # steps between steady-state checks (50 min)
    ss_window: int = 10               # consecutive passing checks required
    ss_tolerance: float = 1e-6
    on_phase_max_time: float = 48 * 3600.0     # generous vs. the observed 20-39h
    off_phase_max_time: float = 150 * 3600.0   # 48h was NOT enough -- 49/50 runs in
                                                # the first k_d_ds sweep hit it as a
                                                # timeout. Even 150h may not be enough
                                                # for every value in a slow-kinetics
                                                # sweep; off_converged in each run's
                                                # .meta.json says whether it actually did.

    def __post_init__(self):
        self.sweep_values = list(self.sweep_values)

        if self.output_dir is None:
            self.output_dir = OUTPUT_ROOT / f"sweep_{self.sweep_parameter}_on_off"
        else:
            self.output_dir = Path(self.output_dir)

        if self.n_processes is None:
            allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
            n_tasks = len(self.sweep_values) * self.n_replicates
            self.n_processes = min(n_tasks, max(1, allocated))


# =============================================================================
# ONE SIMULATION -- the only real rewrite. Same physics, same split solver,
# same validation gates as base.run_single_simulation; the time loop is now
# On_then_off.py's two-phase state machine instead of a fixed 8h.
# =============================================================================

def run_single_simulation(cfg, param_value, replicate_id):
    label = f"{cfg.sweep_parameter}={param_value:g} rep={replicate_id}"
    start_wall = time.perf_counter()

    params = base.apply_sweep_value(cfg, param_value)

    # Resume: identical philosophy to base -- a completed run's timeseries
    # CSV is the durable record, so re-running the script only fills gaps.
    existing = base.timeseries_path_for(cfg, param_value, replicate_id, params)
    if existing.exists():
        print(f"  SKIP {label:<38} already complete ({existing.name})", flush=True)
        return {"param_value": param_value, "replicate_id": replicate_id,
                "success": True, "skipped": True}

    try:
        dt = params["dt"]
        save_every = max(1, int(params["save_interval_time"] / dt))
        epsilon = 1e-10

        # ------------------------------------------------------------- mesh
        mesh_file = base.mesh_path_for(params)
        if not mesh_file.exists():
            raise FileNotFoundError(
                f"Mesh missing: {mesh_file}. It should have been built before "
                f"the pool started."
            )

        mesh = Gmsh2D(str(mesh_file))
        x, y = np.asarray(mesh.cellCenters[0]), np.asarray(mesh.cellCenters[1])

        y_center = params["total_height"] / 2.0
        sender_x = params["total_width"] / 2.0 - params["distance_between"] / 2.0
        receiver_x = params["total_width"] / 2.0 + params["distance_between"] / 2.0

        # ------------------------------------------------------------ fields
        (S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
         sender_mask, receiver_mask) = base.initialize_fields(
            mesh, x, y, sender_x, receiver_x, y_center, params)

        if sender_mask.sum() < 50:
            raise RuntimeError(
                f"Only {sender_mask.sum()} cells inside the sender node. "
                f"Mesh is too coarse or the geometry is wrong.")
        if receiver_mask.sum() < 50:
            raise RuntimeError(
                f"Only {receiver_mask.sum()} cells inside the receiver node.")

        eq_S2 = base.build_S2_equation(
            S2, I2, Th2, I1O2, D_S2,
            k_p=params["k_p"], k_slow=params["k_slow"], k_fast=params["k_fast"],
            k_d_ss=params["k_d_ss"],
        )
        s2_solver = LinearLUSolver(tolerance=1e-10)

        probe_idx = int(np.argmin(np.hypot(x - receiver_x, y - y_center)))

        # ------------------------------------------------------------ storage
        rows = []

        def sample(step, phase):
            rows.append({
                "time_hours": step * dt / 3600.0,
                "I2_center_nM": float(I2.value[probe_idx]) * 1e3,
                "S2_free_center_nM": float(S2.value[probe_idx]) * 1e3,
                "S2_total_center_nM": (
                    float(S2.value[probe_idx])
                    + float(S2_I2.value[probe_idx])
                    + float(S2_Th2.value[probe_idx])) * 1e3,
                "I1O2_active": 1 if phase == "on" else 0,
            })

        sample(0, "on")

        # -------------------------------------------------- two-phase loop
        phase = "on"
        phase_start_time = 0.0
        t_shutoff = None
        t_stop = None
        on_converged = False
        off_converged = False
        recent_changes = []

        step = 0
        current_time = 0.0

        while True:
            S2.updateOld(); I2.updateOld(); Th2.updateOld()
            S2_I2.updateOld(); S2_Th2.updateOld()

            res = 1e10
            n_sweeps = 0
            prev_res = None
            while n_sweeps < base.MAX_SWEEPS:
                S2_guess = S2.value

                I2_new, S2_I2_new = base.reaction_pair_step(
                    S2_guess, I2.old.value, S2_I2.old.value,
                    params["k_slow"], params["k_d_ds"], dt)
                Th2_new, S2_Th2_new = base.reaction_pair_step(
                    S2_guess, Th2.old.value, S2_Th2.old.value,
                    params["k_fast"], params["k_d_ds"], dt)

                I2.setValue(I2_new); S2_I2.setValue(S2_I2_new)
                Th2.setValue(Th2_new); S2_Th2.setValue(S2_Th2_new)

                res = eq_S2.sweep(dt=dt, solver=s2_solver)
                n_sweeps += 1

                if res < base.SWEEP_RESIDUAL_TARGET:
                    break
                if prev_res is not None and abs(res - prev_res) < base.SWEEP_PLATEAU_TOL:
                    break
                prev_res = res

            step += 1
            current_time = step * dt

            if step % save_every == 0:
                sample(step, phase)

            # --- validation gate: sender must be producing S2. Only ever
            # checked this early in the "on" phase (t_shutoff is always well
            # past ss_window*check_interval minutes in), so it can't fire
            # after a deliberate shutoff and be mistaken for a bug.
            if step == max(1, int(600 / dt)):
                if float(np.max(S2.value)) <= 0.0:
                    raise RuntimeError(
                        "No S2 anywhere in the domain after 10 simulated "
                        "minutes. The sender is not transcribing.")

            if step % cfg.check_interval == 0:
                changes = [
                    np.max(np.abs(S2.value[receiver_mask] - S2.old.value[receiver_mask])
                           / (np.abs(S2.value[receiver_mask]) + epsilon)),
                    np.max(np.abs(I2.value[receiver_mask] - I2.old.value[receiver_mask])
                           / (np.abs(I2.value[receiver_mask]) + epsilon)),
                    np.max(np.abs(Th2.value[receiver_mask] - Th2.old.value[receiver_mask])
                           / (np.abs(Th2.value[receiver_mask]) + epsilon)),
                    np.max(np.abs(S2_I2.value[receiver_mask] - S2_I2.old.value[receiver_mask])
                           / (np.abs(S2_I2.value[receiver_mask]) + epsilon)),
                    np.max(np.abs(S2_Th2.value[receiver_mask] - S2_Th2.old.value[receiver_mask])
                           / (np.abs(S2_Th2.value[receiver_mask]) + epsilon)),
                ]
                max_change = np.max(changes)
                recent_changes.append(max_change)
                if len(recent_changes) > cfg.ss_window:
                    recent_changes.pop(0)

                phase_elapsed = current_time - phase_start_time
                phase_cap = (cfg.on_phase_max_time if phase == "on"
                             else cfg.off_phase_max_time)
                converged = (len(recent_changes) >= cfg.ss_window
                             and all(c < cfg.ss_tolerance for c in recent_changes))
                timed_out = phase_elapsed >= phase_cap

                if converged or timed_out:
                    if phase == "on":
                        on_converged = converged
                        t_shutoff = current_time
                        I1O2.setValue(0.0)
                        phase = "off"
                        phase_start_time = current_time
                        recent_changes = []
                    else:
                        off_converged = converged
                        t_stop = current_time
                        break

        df = pd.DataFrame(rows)

        if df["S2_total_center_nM"].max() <= 0.0:
            raise RuntimeError(
                "S2 never reached the receiver node (centre-point value stayed "
                "at exactly zero). This is the signature of a disconnected mesh.")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        ts_file = base.timeseries_path_for(cfg, param_value, replicate_id, params)

        tmp_file = ts_file.with_suffix(".csv.partial")
        df.to_csv(tmp_file, index=False)
        os.replace(tmp_file, ts_file)

        wall = time.perf_counter() - start_wall

        ts_file.with_suffix(".meta.json").write_text(json.dumps({
            "wall_time_s": wall,
            "n_cells": int(mesh.numberOfCells),
            "n_cells_sender_node": int(sender_mask.sum()),
            "n_cells_receiver_node": int(receiver_mask.sum()),
            "mesh_file": mesh_file.name,
            "t_shutoff_hr": t_shutoff / 3600.0,
            "t_stop_hr": t_stop / 3600.0,
            "on_converged": on_converged,
            "off_converged": off_converged,
        }, indent=2))

        final = df.iloc[-1]

        print(f"  OK   {label:<38} "
              f"shutoff={t_shutoff/3600:5.2f}h(conv={on_converged!s:<5}) "
              f"stop={t_stop/3600:6.2f}h(conv={off_converged!s:<5}) "
              f"I2_final={final['I2_center_nM']:7.2f} nM  "
              f"[{wall/60:.1f} min]", flush=True)

        return {
            "param_value": param_value,
            "replicate_id": replicate_id,
            "I2_center_final_nM": final["I2_center_nM"],
            "S2_free_center_final_nM": final["S2_free_center_nM"],
            "S2_total_center_final_nM": final["S2_total_center_nM"],
            "half_time_center_hr": base.half_time(df["time_hours"].values,
                                                   df["I2_center_nM"].values),
            "t_shutoff_hr": t_shutoff / 3600.0,
            "t_stop_hr": t_stop / 3600.0,
            "on_converged": on_converged,
            "off_converged": off_converged,
            "n_cells": int(mesh.numberOfCells),
            "wall_time_s": wall,
            "timeseries_file": str(ts_file),
            "success": True,
        }

    except Exception as exc:
        import traceback
        print(f"  FAIL {label:<38} {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "param_value": param_value,
            "replicate_id": replicate_id,
            "wall_time_s": time.perf_counter() - start_wall,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# =============================================================================
# ORCHESTRATION -- has to call the run_single_simulation above, so this can't
# be reused from base as-is (base.run_sweep is bound to base's own worker).
# =============================================================================

def run_sweep(cfg):
    tasks = [(cfg, value, rep)
             for value in cfg.sweep_values
             for rep in range(cfg.n_replicates)]

    print(f"\nRunning {len(tasks)} simulation(s) on {cfg.n_processes} process(es).")
    print(f"Each run turns I1O2 off adaptively (receiver steady state, "
          f"tolerance={cfg.ss_tolerance:g}, or a {cfg.on_phase_max_time/3600:.0f}h "
          f"cap), then continues until the receiver re-converges or hits a "
          f"{cfg.off_phase_max_time/3600:.0f}h cap.\n")

    t0 = time.time()
    if cfg.n_processes == 1:
        results = [run_single_simulation(*task) for task in tasks]
    else:
        with Pool(processes=cfg.n_processes) as pool:
            results = pool.starmap(run_single_simulation, tasks)

    print(f"\nAll simulations finished in {(time.time() - t0) / 60:.1f} min.")
    return results


def plot_timeseries(cfg):
    """
    Overlay every run's I2 timeseries, with a downward-triangle marker at
    each run's own shutoff point (reads t_shutoff_hr back from the .meta.json
    sidecar written by run_single_simulation). Deliberately just this one
    panel -- see module docstring on why no new metrics yet.
    """
    files = sorted(cfg.output_dir.glob("timeseries_*.csv"))
    if not files:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cfg.sweep_values)))

    for color, value in zip(colors, cfg.sweep_values):
        matches = sorted(cfg.output_dir.glob(
            f"timeseries_{cfg.sweep_parameter}={value:g}_rep=*.csv"))
        for path in matches:
            df = pd.read_csv(path)
            label = f"{cfg.sweep_parameter}={value:g}"
            ax.plot(df["time_hours"], df["I2_center_nM"],
                    color=color, lw=1.6, label=label)

            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                t_shut = meta.get("t_shutoff_hr")
                if t_shut is not None:
                    idx = (df["time_hours"] - t_shut).abs().idxmin()
                    ax.scatter([df["time_hours"][idx]], [df["I2_center_nM"][idx]],
                               color=color, marker="v", s=45, zorder=5,
                               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("[I2] at receiver (nM)")
    ax.set_title(f"On/off timeseries: {cfg.sweep_parameter}\n(▼ = I1O2 shutoff)")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)

    fig.tight_layout()
    out = cfg.output_dir / f"timeseries_on_off_{cfg.sweep_parameter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


# =============================================================================
# ENTRY POINT -- call this from each per-parameter sweep script
# =============================================================================

def run(cfg):
    print("=" * 78)
    print("ON/OFF PARAMETER SWEEP")
    print("=" * 78)
    print(f"parameter : {cfg.sweep_parameter}")
    print(f"values    : {cfg.sweep_values}")
    print(f"replicates: {cfg.n_replicates}")
    print(f"output    : {cfg.output_dir}")
    print(f"phase ctrl: check_interval={cfg.check_interval} steps, "
          f"ss_window={cfg.ss_window}, ss_tolerance={cfg.ss_tolerance:g}, "
          f"on_phase_max_time={cfg.on_phase_max_time/3600:.0f}h, "
          f"off_phase_max_time={cfg.off_phase_max_time/3600:.0f}h")
    print("=" * 78)

    if cfg.sweep_parameter not in base.DEFAULT_PARAMS:
        raise SystemExit(
            f"'{cfg.sweep_parameter}' is not a known parameter. "
            f"Valid names:\n  {sorted(base.DEFAULT_PARAMS)}")

    print("\nSTAGE 1 - meshes\n")
    base.build_all_meshes(cfg)

    print("\nSTAGE 2 - simulations\n")
    results = run_sweep(cfg)

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} simulation(s) FAILED this session:")
        for r in failed:
            print(f"  {cfg.sweep_parameter}={r['param_value']}: "
                  f"{r.get('error', 'unknown error')}")
        print("Completed runs are unaffected and are summarised below.")

    print("\nSTAGE 3 - analysis\n")
    stats = base.summarise(cfg)
    base.plot(cfg, stats)
    plot_timeseries(cfg)

    print("\nDone.\n")
