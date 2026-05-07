#!/usr/bin/env python3
"""
exp3_chained_replanning.py
==========================
Experiment 3: Multi-Step Replanning Chains (Single-Failure-Per-Checkpoint)

Goal
----
Show that the replanner can be called REPEATEDLY across an unfolding mission
without compounding errors, losing state, or getting stuck in loops.

Each checkpoint injects AT MOST ONE failure signal.  Compound / simultaneous
failures are deliberately reserved for Experiment 4, so that the two
experiments isolate distinct phenomena:

  Exp 3 — Can the replanner sustain correctness across a chain?
  Exp 4 — Can the replanner correctly PRIORITISE when signals conflict?

Design Notes on Step Lifecycle
--------------------------------
A critical modelling distinction:

  AdvanceSteps(n)            — steps completed SUCCESSFULLY; moved to history.
  MarkCurrentStepFailed()    — the step currently executing produced invalid
                               output and was aborted. It is REMOVED from
                               remaining_steps and NOT added to completed_steps.
                               The replanner therefore sees a gap: the step
                               is neither in history (it did not succeed) nor
                               in the remaining tail (it was consumed). The
                               failure_context explains what went wrong in
                               factual terms only — it does NOT tell the
                               replanner what corrective steps to take. That
                               inference is what the experiment measures.

Without this distinction every mid-execution failure becomes invisible: the
replanner sees the step already in history, the remaining tail looks correct,
and it says NOMINAL — a false negative.

Why the gap matters for the replanner
--------------------------------------
When MarkCurrentStepFailed pops the step, the world state presented to the
replanner has:
  - completed_steps: steps that SUCCEEDED (including any approach that ran
                     before the failed map)
  - remaining_steps: everything scheduled AFTER the failed step (e.g.
                     [return_home] if the failed step was the last map)
  - cylinders:       the full discovered manifest

The replanner must notice that cylinder N appears in the manifest but has
zero completed map passes, that the remaining tail has no map(N) step, and
therefore must insert approach(N)+map(N) before return_home. This is the
inference we are testing — the failure_context supplies only the factual
cause (e.g. "IMU dropout during orbit of cylinder 1"), not the remedy.

Design Notes on Ambiguity
--------------------------
A key design constraint: every scenario that expects the replanner to REPLAN
must have a failure_context that provides enough factual signal to diagnose
the gap — without prescribing the fix. Scenarios where the correct answer
could be argued either way are resolved to NOMINAL (conservative baseline).

Reference Tail Plans
--------------------
Every checkpoint carries an `expected_tail` alongside `expected_nominal`:
  - If expected_nominal is True:  expected_tail must be None.
  - If expected_nominal is False: expected_tail is a list[dict] defining the
    EXACT correct tail the replanner should produce.

Tail matching uses _tails_match(), which checks:
  - Same number of steps
  - Each step matches on "state", "args" keys (cylinder_id, mode, standoff_distance,
    min_altitude_m where present), and "repeat"

This gives us a single, unambiguous correctness oracle for every checkpoint.
There is no longer any separate "tail content validator" callback — the
reference tail IS the content check.

Scoring
-------
For confusion matrix purposes, the decision (NOMINAL vs REPLAN) is the signal:

  True Positive  (TP): Expected REPLAN, replanner said REPLAN
  False Positive (FP): Expected NOMINAL, replanner said REPLAN   ← spurious change
  True Negative  (TN): Expected NOMINAL, replanner said NOMINAL  ← correct quiescence
  False Negative (FN): Expected REPLAN, replanner said NOMINAL   ← missed failure

behavior_accuracy_pct (PRIMARY METRIC) requires:
  - For expected NOMINAL checkpoints:  replanner must say NOMINAL            (TN)
  - For expected REPLAN checkpoints:   replanner must say REPLAN  AND  the
    produced tail must match the reference tail exactly             (TP + correct tail)

decision_accuracy_pct measures only the binary NOMINAL/REPLAN choice (TP+TN rate),
ignoring tail content. Comparing decision_accuracy vs behavior_accuracy isolates
whether errors come from wrong decisions or wrong tail content.

Chains (11 total across 4 complexity tiers):
  Simple   (3):  1 cylinder, 1 failure per chain
  Medium   (3):  2 cylinders, 2-3 sequential single failures
  Complex  (3):  3 cylinders, varied failure modes and heterogeneous missions
  Long     (2):  end-to-end chains with no failures (regression / NOMINAL check)
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from experiment_utils import (
    MODELS,
    MockCylinder,
    MockWorldState,
    _extract_mapped_cylinders,
    fmt_plan,
    run_replan,
    save_json,
    validate_plan,
)

# ---------------------------------------------------------------------------
# EVENT TYPES
# ---------------------------------------------------------------------------


@dataclass
class AdvanceSteps:
    """Move N steps from remaining -> completed (simulates SUCCESSFUL execution)."""
    n: int = 1


@dataclass
class MarkCurrentStepFailed:
    """
    The step currently executing aborted with invalid output.
    The step is REMOVED from remaining_steps and NOT added to completed_steps.
    failure_context must be factual-only — no corrective prescription.
    """
    context: str


@dataclass
class InjectFailure:
    """
    Replace failure_context with a new string representing an EXTERNAL event.
    The step lists are not changed.
    """
    context: str


@dataclass
class ClearFailure:
    """Clear failure_context after the replanner has handled it."""
    pass


@dataclass
class InjectCylinder:
    """Add a newly-discovered cylinder to the world manifest."""
    cylinder: MockCylinder


@dataclass
class UpdateBattery:
    """Update the battery reading."""
    pct: float


Event = AdvanceSteps | MarkCurrentStepFailed | InjectFailure | ClearFailure | InjectCylinder | UpdateBattery


# ---------------------------------------------------------------------------
# HELPER BUILDERS
# ---------------------------------------------------------------------------


def _step(state: str, args: dict | None = None, repeat: int = 1) -> dict:
    return {"state": state, "args": args or {}, "repeat": repeat}


def _takeoff(alt: float = 5.0) -> dict:
    return _step("takeoff", {"altitude": alt})


def _search(pattern: str = "yaw_scan") -> dict:
    return _step("search", {"pattern": pattern})


def _approach(cid: int, sd: float = 5.0) -> dict:
    return _step("approach", {"cylinder_id": cid, "standoff_distance": sd})


def _map_orbit(cid: int, sd: float = 5.0, repeat: int = 1) -> dict:
    return _step("map", {"cylinder_id": cid, "standoff_distance": sd, "mode": "orbit"}, repeat)


def _map_vmap(cid: int, sd: float = 5.0, repeat: int = 1) -> dict:
    return _step(
        "map",
        {"cylinder_id": cid, "standoff_distance": sd,
         "mode": "vertical_map", "min_altitude_m": 2.0},
        repeat,
    )


def _return_home() -> dict:
    return _step("return_home")


def _cyl(id_: int, x: float, y: float = 0.0, depth: float = 5.0) -> MockCylinder:
    return MockCylinder(id=id_, world_x=x, world_y=y, depth_m=depth)


# ---------------------------------------------------------------------------
# TAIL MATCHING
# ---------------------------------------------------------------------------

# Fields compared per step when matching a produced tail against a reference tail.
# "repeat" is always compared. Fields in _ARGS_KEYS are compared inside "args"
# only when they appear in the REFERENCE step — extra args keys in the produced
# tail that are absent from the reference are ignored (permissive on extras).
_ARGS_KEYS = ("cylinder_id", "mode", "standoff_distance", "min_altitude_m")


def _steps_match(produced: dict, reference: dict) -> bool:
    """
    Return True if a produced step matches a reference step.

    Matching rules:
      - "state" must be identical.
      - "repeat" must be identical (defaults to 1 if absent).
      - For each args key present in the REFERENCE step, the produced step
        must carry the same value. Extra keys in the produced step that are
        absent from the reference are silently allowed (e.g. a model that
        adds "altitude" to return_home would still match a reference that
        has no args).
      - Numeric args are compared as floats to avoid int/float mismatches.
    """
    if produced.get("state") != reference.get("state"):
        return False
    if produced.get("repeat", 1) != reference.get("repeat", 1):
        return False
    ref_args  = reference.get("args", {})
    prod_args = produced.get("args", {})
    for key in _ARGS_KEYS:
        if key not in ref_args:
            continue   # reference doesn't constrain this key
        ref_val  = ref_args[key]
        prod_val = prod_args.get(key)
        # Numeric tolerance for float fields (standoff_distance, min_altitude_m).
        if isinstance(ref_val, (int, float)) and isinstance(prod_val, (int, float)):
            if abs(float(ref_val) - float(prod_val)) > 1e-6:
                return False
        else:
            if ref_val != prod_val:
                return False
    return True


def _tails_match(produced: list[dict] | None, reference: list[dict] | None) -> tuple[bool, str]:
    """
    Compare a produced tail against the reference tail.

    Returns (match: bool, explanation: str).

    Matching is structural: same length, each step satisfies _steps_match().
    The reference tail is the ground truth stored in ChainScenario.expected_tails.

    Special cases:
      - If reference is None the checkpoint is unscored (returns True, note).
      - If produced is None (NOMINAL decision) and reference is not None this
        is always a mismatch (returns False, explanation).
    """
    if reference is None:
        return True, "No reference tail — checkpoint is unscored."
    if produced is None:
        return False, "Replanner said NOMINAL but a reference replan tail exists."
    if len(produced) != len(reference):
        return (
            False,
            f"Tail length mismatch: produced {len(produced)} steps, "
            f"reference has {len(reference)} steps.",
        )
    mismatches = []
    for i, (p, r) in enumerate(zip(produced, reference)):
        if not _steps_match(p, r):
            mismatches.append(
                f"  step[{i}]: produced {p!r} ≠ reference {r!r}"
            )
    if mismatches:
        return False, "Step mismatch(es):\n" + "\n".join(mismatches)
    return True, "Tail matches reference exactly."


# ---------------------------------------------------------------------------
# CHAIN SCENARIO
# ---------------------------------------------------------------------------


@dataclass
class ChainScenario:
    id: str
    complexity: str          # "simple" | "medium" | "complex" | "long"
    mission: str
    initial_world: MockWorldState
    events: list[list]       # one inner list per checkpoint
    note: str

    # Per-checkpoint ground truth.
    #
    # expected_nominal[i]:
    #   True  — replanner should say NOMINAL at checkpoint i
    #   False — replanner should say REPLAN at checkpoint i
    #   None  — checkpoint is unscored (no expected value provided)
    expected_nominal: list[bool | None] = field(default_factory=list)

    # expected_tails[i]:
    #   None           — no reference tail (either expected NOMINAL, or unscored)
    #   list[dict]     — the EXACT correct tail the replanner must produce when
    #                    expected_nominal[i] is False.  _tails_match() is used
    #                    to compare the produced tail against this reference.
    #
    # Invariant: if expected_nominal[i] is True, expected_tails[i] must be None.
    expected_tails: list[list[dict] | None] = field(default_factory=list)

    # Per-checkpoint human-readable rationale (for printout / paper annotation).
    expected_reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CHAIN DEFINITIONS
# ---------------------------------------------------------------------------
#
# Reference tails (expected_tails) encode the EXACT correct replanner output.
# They are compared step-by-step via _tails_match().
#
# Notation used in comments below:
#   approach(N)     → _approach(N)  with standoff_distance=5.0 (default)
#   map_orbit(N)    → _map_orbit(N) with mode="orbit", repeat=1 (default)
#   map_vmap(N,r)   → _map_vmap(N)  with mode="vertical_map", repeat=r
#   return_home()   → _return_home()
# ---------------------------------------------------------------------------

CHAINS: list[ChainScenario] = [

    # =========================================================================
    # SIMPLE — 1 cylinder, single failure, tests basic gap detection
    # =========================================================================

    ChainScenario(
        id="S1",
        complexity="simple",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0)],
            completed_steps=[_takeoff(), _search()],
            remaining_steps=[_approach(1), _map_orbit(1), _return_home()],
        ),
        events=[
            # cp0 — no events; plan already covers cyl(1) correctly.
            [],

            # cp1 — approach(1) succeeds; map(1) aborts mid-arc (IMU dropout).
            #        MarkCurrentStepFailed pops map(1). Gap: cyl(1) in manifest
            #        with 0 passes; remaining=[return_home] with no map(1). REPLAN.
            [AdvanceSteps(1),
             MarkCurrentStepFailed(
                 "Map orbit of cylinder 1 aborted mid-arc: IMU dropout. "
                 "Recorded point-cloud data is unusable."
             )],

            # cp2 — replanner rebuilt tail; all three steps advance cleanly. NOMINAL.
            [ClearFailure(), AdvanceSteps(3)],
        ],
        expected_nominal=[True, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — re-approach + re-map + return_home.
            [_approach(1), _map_orbit(1), _return_home()],
            # cp2: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Plan fully covers cylinder 1 (approach→map→home) with no failure context. NOMINAL.",
            "approach(1) succeeded; map(1) aborted (IMU dropout, unusable data) and was "
            "popped from tail. Cylinder 1 is in manifest with 0 completed passes; remaining "
            "=[return_home] has no map(1). Gap is unambiguous — replanner must insert "
            "approach(1)+map(1) before return_home without being told to do so. REPLAN.",
            "Replanner rebuilt tail; all steps (re-approach, re-map, return_home) advanced "
            "cleanly. Chain converged. NOMINAL.",
        ],
        note=(
            "Single cylinder. approach(1) succeeds; map(1) aborts mid-arc (IMU dropout, "
            "data unusable). MarkCurrentStepFailed pops map(1) — replanner sees cylinder "
            "1 in manifest with 0 passes and [return_home] in tail with no map(1). Must "
            "insert approach(1)+map(1) before return_home by inferring the gap from "
            "manifest vs. history. Failure context is factual only."
        ),
    ),

    ChainScenario(
        id="S2",
        complexity="simple",
        mission="Find any cylinders and orbit each one, then return home.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 7.0)],
            completed_steps=[_takeoff(), _search(), _approach(1)],
            remaining_steps=[_map_orbit(1), _return_home()],
        ),
        events=[
            # cp0 — map(1) succeeds; new cylinder detected on return leg. REPLAN.
            [AdvanceSteps(1),
             InjectCylinder(_cyl(2, 14.0)),
             InjectFailure(
                 "New cylinder detected at approximately 14 m during return_home."
             )],

            # cp1 — replanner extended tail; both cylinders handled. NOMINAL.
            [ClearFailure(), AdvanceSteps(3)],
        ],
        expected_nominal=[False, True],
        expected_tails=[
            # cp0: REPLAN — extend tail with approach(2)+map(2)+return_home.
            [_approach(2), _map_orbit(2), _return_home()],
            # cp1: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "map(1) succeeded; cylinder 2 injected into manifest with 0 passes and no "
            "approach(2)/map(2) scheduled. Mission requires orbiting every cylinder. "
            "Remaining=[return_home] — cannot satisfy mission without extending tail. REPLAN.",
            "Replanner extended tail with approach(2)+map(2)+home; all steps advanced. "
            "Both cylinders mapped. Chain converged. NOMINAL.",
        ],
        note=(
            "Cylinder 1 mapped cleanly. New cylinder (cyl 2) spotted on the return leg. "
            "InjectFailure supplies only the factual detection event at ~14 m — no "
            "instruction about what steps to add. Replanner must infer approach(2)+map(2) "
            "is needed before return_home."
        ),
    ),

    ChainScenario(
        id="S3",
        complexity="simple",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 6.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1)],
            remaining_steps=[_return_home()],
        ),
        events=[
            # cp0 — on return leg; high-confidence new cylinder detected. REPLAN.
            [InjectCylinder(_cyl(2, 12.0)),
             InjectFailure(
                 "High-confidence detection of an untracked cylinder at approximately "
                 "12 m."
             )],

            # cp1 — failure cleared; replanner added approach(2)+map(2)+home. NOMINAL.
            [ClearFailure()],

            # cp2 — approach(2)+map(2) advance; return_home remains. NOMINAL.
            [AdvanceSteps(2)],
        ],
        expected_nominal=[False, True, True],
        expected_tails=[
            # cp0: REPLAN — add approach(2)+map(2) before return_home.
            [_approach(2), _map_orbit(2), _return_home()],
            # cp1: NOMINAL — no reference tail.
            None,
            # cp2: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Cylinder 2 injected into manifest with 0 passes; remaining=[return_home] "
            "has no approach(2)/map(2). Mission requires mapping every found cylinder. REPLAN.",
            "Replanner extended tail; failure cleared. Remaining plan now correctly covers "
            "cylinder 2. No further action needed. NOMINAL.",
            "approach(2) and map(2) advanced cleanly; return_home remains. Quiescent. NOMINAL.",
        ],
        note=(
            "Cylinder 1 already mapped. New cylinder (cyl 2) discovered while on the "
            "return leg after the original plan was considered complete. InjectFailure "
            "supplies factual detection event at ~12 m only — no prescription. Tests "
            "whether replanner extends a near-complete plan for a late discovery."
        ),
    ),

    # =========================================================================
    # MEDIUM — 2 cylinders, 2-3 sequential single failures
    # =========================================================================

    ChainScenario(
        id="M1",
        complexity="medium",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search()],
            remaining_steps=[
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
                _return_home(),
            ],
        ),
        events=[
            # cp0 — NOMINAL — plan covers both cylinders correctly.
            [],

            # cp1 — cyl1+cyl2 approach sequence executes cleanly; visual contact
            #        with cyl2 lost after approach(2) before mapping. REPLAN.
            #        Remaining after AdvanceSteps(3): [map(2), return_home].
            #        Visual contact lost → approach must be re-inserted before map(2).
            [AdvanceSteps(3),
             InjectFailure(
                 "Visual contact with cylinder 2 lost after approach before planned "
                 "mapping. Relative position to cylinder 2 is unreliable."
             )],

            # cp2 — replanner inserted re-approach(2); cyl3 spotted on return. REPLAN.
            #        After ClearFailure+AdvanceSteps(2): approach(2_retry)+map(2) done.
            #        Cylinder 3 detected — no approach(3)/map(3) in tail. REPLAN.
            [ClearFailure(), AdvanceSteps(2),
             InjectCylinder(_cyl(3, 16.0)),
             InjectFailure(
                 "High-confidence detection of untracked cylinder at approximately "
                 "16 m."
             )],

            # cp3 — all three cylinders handled; chain converges. NOMINAL.
            [ClearFailure(), AdvanceSteps(3)],
        ],
        expected_nominal=[True, False, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — re-insert approach(2) before the still-scheduled map(2).
            #      Remaining before this replan: [map(2), return_home].
            #      Correct fix: [approach(2), map(2), return_home].
            [_approach(2), _map_orbit(2), _return_home()],
            # cp2: REPLAN — extend tail for cylinder 3.
            #      Remaining before this replan: [return_home] (after replanner's
            #      approach(2)+map(2) advanced in AdvanceSteps(2)).
            #      Correct fix: [approach(3), map(3), return_home].
            [_approach(3), _map_orbit(3), _return_home()],
            # cp3: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Both cylinders covered with correct approach→map pairs. No failure. NOMINAL.",
            "approach(2) completed but visual contact immediately lost before mapping. "
            "map(2) is in tail but standoff geometry is invalid — replanner must insert "
            "approach(2) before existing map(2) to re-establish position. REPLAN.",
            "approach(2) re-established; map(2) completed. Cylinder 3 detected at ~16 m "
            "with 0 passes and no approach(3)/map(3) in tail. Mission requires mapping "
            "every found cylinder. REPLAN.",
            "All three cylinders handled; approach(3)+map(3)+home advanced cleanly. NOMINAL.",
        ],
        note=(
            "FAILURE 1 (InjectFailure): approach(2) completes but visual contact with "
            "cylinder 2 is immediately lost before any scan — map(2) remains in tail but "
            "standoff is invalid; replanner must insert re-approach before existing map(2). "
            "FAILURE 2 (InjectCylinder + InjectFailure): cylinder 3 detected at ~16 m on "
            "return; replanner must extend tail with approach(3)+map(3). Both failure "
            "contexts are factual-only with no corrective prescription."
        ),
    ),

    ChainScenario(
        id="M2",
        complexity="medium",
        mission="Do a thorough search then do 2 vertical mapping laps of each cylinder found.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 6.0)],
            completed_steps=[_takeoff(6.0), _search("lawnmower")],
            remaining_steps=[
                _approach(1, sd=5.0),
                _map_vmap(1, sd=5.0, repeat=1),   # lap 1
                _map_vmap(1, sd=5.0, repeat=1),   # lap 2
                _return_home(),
            ],
        ),
        events=[
            # cp0 — NOMINAL — plan covers 2 vmap laps correctly.
            [],

            # cp1 — approach(1) succeeds; vmap lap 1 aborts (IMU spike). REPLAN.
            #        MarkCurrentStepFailed pops lap1. Remaining: [lap2, return_home].
            #        Mission requires 2 laps; only 1 remains with 0 done → shortfall.
            #        Correct fix: re-approach + lap1_retry + lap2 + return_home.
            [AdvanceSteps(1),
             MarkCurrentStepFailed(
                 "Vertical map lap 1 of cylinder 1 aborted: unrecoverable IMU spike "
                 "after first arc segment. Lap 1 data is unusable."
             )],

            # cp2 — replanner rebuilt 2-lap sequence; re-approach+lap1 advance.
            #        Centroid drift 0.4 m is WITHIN 0.6 m tolerance. NOMINAL.
            [ClearFailure(), AdvanceSteps(2),
             InjectFailure(
                 "Cylinder 1 centroid drifted 0.4 m during lap 2 vertical sweep. "
                 "Drift is within the accepted 0.6 m tolerance."
             )],

            # cp3 — lap2+return_home advance cleanly. NOMINAL.
            [ClearFailure(), AdvanceSteps(2)],
        ],
        expected_nominal=[True, False, True, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — rebuild 2-lap sequence from scratch.
            #      Remaining before replan: [lap2, return_home].
            #      Correct fix: [approach(1), lap1, lap2, return_home].
            [_approach(1, sd=5.0), _map_vmap(1, sd=5.0, repeat=1),
             _map_vmap(1, sd=5.0, repeat=1), _return_home()],
            # cp2: NOMINAL (centroid drift within tolerance) — no reference tail.
            None,
            # cp3: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Plan covers approach(1) + 2 separate vmap laps + return_home. NOMINAL.",
            "approach(1) succeeded; lap 1 aborted (IMU spike, data unusable) and popped "
            "from tail. Only lap 2 remains but mission requires 2 completed laps (0 done). "
            "Replanner must infer the shortfall and insert approach(1)+lap1_retry before "
            "existing lap2. REPLAN.",
            "Replanner rebuilt 2-lap sequence; re-approach + lap1_retry succeeded. Lap 2 "
            "underway — centroid drift 0.4 m is explicitly within 0.6 m tolerance. "
            "Remaining plan (lap2+home) is still correct. Replanner must NOT change it. "
            "NOMINAL.",
            "lap2 and return_home advanced cleanly. Chain converged. NOMINAL.",
        ],
        note=(
            "Mission demands 2 vertical laps. FAILURE 1: lap 1 aborts mid-arc (IMU spike, "
            "data unusable), popped from tail — only 1 lap remains; replanner must detect "
            "the shortfall and reconstruct the 2-lap sequence (REPLAN). "
            "FAILURE 2: soft centroid-drift warning at 0.4 m, explicitly within 0.6 m "
            "tolerance — replanner must NOT replan (NOMINAL). Directly tests discrimination "
            "between actionable failures and informational soft warnings."
        ),
    ),

    ChainScenario(
        id="M3",
        complexity="medium",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1)],
            remaining_steps=[_approach(2), _map_orbit(2), _return_home()],
        ),
        events=[
            # cp0 — NOMINAL — remaining plan correctly covers cylinder 2.
            [],

            # cp1 — approach(2) succeeds; map(2) aborted (camera failure, all frames
            #        corrupted). MarkCurrentStepFailed pops map(2). REPLAN.
            [AdvanceSteps(1),
             MarkCurrentStepFailed(
                 "Map orbit of cylinder 2 aborted: camera shutter failure. "
                 "All frames corrupted. Cylinder 2 has 0 valid mapping passes."
             )],

            # cp2 — replanner rebuilt approach(2)+map(2)+home. New approach(2)
            #        aborts (wind gust drift). map(2) in tail with no approach. REPLAN.
            [ClearFailure(),
             MarkCurrentStepFailed(
                 "Approach to cylinder 2 aborted: Standoff geometry was not established."
             )],

            # cp3 — replanner inserted approach(2) before map(2); all advance. NOMINAL.
            [ClearFailure(), AdvanceSteps(3)],
        ],
        expected_nominal=[True, False, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — re-schedule approach(2)+map(2)+return_home.
            [_approach(2), _map_orbit(2), _return_home()],
            # cp2: REPLAN — approach(2) retry aborted; map(2) still in tail but has
            #      no preceding approach. Must insert approach(2) before map(2).
            #      Remaining before replan: [map(2), return_home].
            #      Correct fix: [approach(2), map(2), return_home].
            [_approach(2), _map_orbit(2), _return_home()],
            # cp3: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Cylinder 1 mapped; cylinder 2 has correct approach→map scheduled. NOMINAL.",
            "approach(2) succeeded; map(2) aborted (camera failure, all frames corrupted, "
            "0 valid passes) and popped from tail. Cylinder 2 in manifest with 0 passes; "
            "remaining=[return_home] has no map(2). Replanner must insert "
            "approach(2)+map(2) before return_home. REPLAN.",
            "approach(2) retry aborted (wind gust drift, standoff not established) "
            "and popped from tail. map(2) remains in tail but has no preceding approach — "
            "constraint violation and invalid geometry. Replanner must insert approach(2) "
            "before existing map(2). REPLAN.",
            "approach(2) and map(2) executed cleanly; return_home advanced. NOMINAL.",
        ],
        note=(
            "FAILURE 1: map(2) aborted (camera failure, all frames corrupted). "
            "FAILURE 2: replanner-inserted approach(2) retry aborts (wind gust). "
            "Two consecutive MarkCurrentStepFailed events test sustained constraint "
            "awareness across a chain."
        ),
    ),

    # =========================================================================
    # COMPLEX — varied failure modes and heterogeneous mission types
    #
    #   C1 — Sequential late-discovery: two NEW cylinders spotted one-at-a-time
    #        on the return leg after cylinder 1 is already fully mapped.
    #        Replanner must extend the tail incrementally for each discovery.
    #
    #   C2 — Abort-policy adherence: user mission explicitly states "if approach
    #        OR mapping of any cylinder fails, skip it and return home."
    #        Tests whether the replanner honours user-specified abort semantics
    #        rather than defaulting to a retry.
    #
    #   C3 — Heterogeneous per-cylinder instructions: the mission specifies a
    #        DIFFERENT action for each discovered cylinder (orbit x2 / vmap /
    #        approach-only). Cylinders discovered one-at-a-time; replanner must
    #        match each to the correct step type.
    # =========================================================================

    ChainScenario(
        id="C1",
        complexity="complex",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0)],
            completed_steps=[_takeoff(), _search("lawnmower")],
            remaining_steps=[_approach(1), _map_orbit(1), _return_home()],
        ),
        events=[
            # ------------------------------------------------------------------
            # cp0: NOMINAL — plan covers cylinder 1 correctly.
            # ------------------------------------------------------------------
            [],

            # ------------------------------------------------------------------
            # cp1: approach(1)+map(1) complete (AdvanceSteps(2)).
            #      Cylinder 2 detected at ~11 m during return leg.
            #      remaining=[return_home]; no approach(2)/map(2) scheduled. REPLAN.
            # ------------------------------------------------------------------
            [AdvanceSteps(2),
             InjectCylinder(_cyl(2, 11.0)),
             InjectFailure(
                 "High-confidence detection of untracked cylinder at approximately "
                 "11 m during return leg."
             )],

            # ------------------------------------------------------------------
            # cp2: Failure cleared; replanner extended tail with approach(2)+map(2)+home.
            #      While navigating toward cylinder 2, cylinder 3 detected at ~17 m.
            #      Current tail: [approach(2), map(2), return_home] — no cyl3. REPLAN.
            # ------------------------------------------------------------------
            [ClearFailure(),
             InjectCylinder(_cyl(3, 17.0)),
             InjectFailure(
                 "High-confidence detection of untracked cylinder at approximately "
                 "17 m during approach to cylinder 2."
             )],

            # ------------------------------------------------------------------
            # cp3: Failure cleared; AdvanceSteps(4) advances
            #      approach(2)+map(2)+approach(3)+map(3) → history.
            #      return_home is the only remaining step. NOMINAL.
            # ------------------------------------------------------------------
            [ClearFailure(), AdvanceSteps(4)],
        ],
        expected_nominal=[True, False, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — extend with approach(2)+map(2)+return_home.
            [_approach(2), _map_orbit(2), _return_home()],
            # cp2: REPLAN — keep cyl2 sequence intact; append cyl3 before return_home.
            #      Current tail before replan: [approach(2), map(2), return_home].
            #      Correct fix: [approach(2), map(2), approach(3), map(3), return_home].
            [_approach(2), _map_orbit(2), _approach(3), _map_orbit(3), _return_home()],
            # cp3: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Plan covers cylinder 1 with approach→map→home. No failure. NOMINAL.",
            "Cylinder 1 mapped; cylinder 2 detected at ~11 m with 0 passes and no "
            "approach(2)/map(2) in remaining=[return_home]. Mission requires mapping "
            "every found cylinder. Replanner must extend tail. REPLAN.",
            "Cylinder 3 detected at ~17 m with 0 passes; current tail handles cylinder 2 "
            "but has no approach(3)/map(3). Replanner must insert cylinder 3 sequence "
            "while preserving existing cylinder 2 steps. REPLAN.",
            "All three cylinders handled; approach(2)+map(2)+approach(3)+map(3) advanced "
            "cleanly. return_home remains. NOMINAL.",
        ],
        note=(
            "C1: Sequential late-discovery — cylinder 1 fully mapped before any new "
            "discovery. Two additional cylinders (cyl 2 and cyl 3) are each detected "
            "one-at-a-time on the return/navigation leg. At cp2 cyl 2 is already "
            "scheduled but not yet executed when cyl 3 is discovered — replanner must "
            "extend the tail for cyl 3 WITHOUT disturbing the existing cyl 2 sequence. "
            "Both failure contexts are factual-only detection events."
        ),
    ),

    ChainScenario(
        id="C2",
        complexity="complex",
        mission=(
            "Search for cylinders and map every one you find. "
            "If approach or mapping of any cylinder fails for any reason, "
            "skip that cylinder and return home immediately. Do not retry."
        ),
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search()],
            remaining_steps=[
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
                _return_home(),
            ],
        ),
        events=[
            # ------------------------------------------------------------------
            # cp0: NOMINAL — plan covers both cylinders. No failure. NOMINAL.
            # ------------------------------------------------------------------
            [],

            # ------------------------------------------------------------------
            # cp1: approach(1)+map(1) execute cleanly (AdvanceSteps(2)).
            #      approach(2) aborts: no-fly zone boundary, standoff never established.
            #      MarkCurrentStepFailed pops approach(2).
            #      remaining=[map(2), return_home].
            #
            #      Mission abort policy: "if approach OR mapping fails, skip and
            #      return home immediately. Do not retry."
            #      Correct tail: [return_home] only — no approach(2), no map(2).
            # ------------------------------------------------------------------
            [AdvanceSteps(2),
             MarkCurrentStepFailed(
                 "Approach to cylinder 2 aborted: UAV entered no-fly zone boundary "
                 "before reaching standoff. Standoff was never established."
             )],

            # ------------------------------------------------------------------
            # cp2: Failure cleared; replanner returned [return_home].
            #      AdvanceSteps(1) advances return_home → history. NOMINAL.
            # ------------------------------------------------------------------
            [ClearFailure(), AdvanceSteps(1)],
        ],
        expected_nominal=[True, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — abort policy → tail must be exactly [return_home].
            [_return_home()],
            # cp2: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "Both cylinders covered with correct approach→map pairs. No failure. NOMINAL.",
            "approach(2) aborted (no-fly zone, standoff never established) and popped "
            "from tail. remaining=[map(2), return_home]. "
            "User mission explicitly states: skip the cylinder and return home immediately "
            "on any failure. Do not retry. Replanner must NOT re-schedule approach(2) "
            "or attempt map(2). Correct tail: [return_home]. REPLAN.",
            "Replanner produced [return_home]; step advanced. Chain converged. NOMINAL.",
        ],
        note=(
            "C2: Abort-policy adherence. User mission explicitly says: if approach OR "
            "mapping of any cylinder fails, skip it and return home immediately — no "
            "retries. Cylinder 1 maps cleanly. approach(2) aborts (no-fly zone). "
            "The replanner must read the mission text and respond with [return_home] "
            "rather than defaulting to a re-approach retry. "
            "Reference tail [return_home] enforces that map(2)/approach(2) are absent."
        ),
    ),

    ChainScenario(
        id="C3",
        complexity="complex",
        mission=(
            "Search for cylinders. If you find cylinders, orbit the first cylinder twice. "
            "If you find a second cylinder, perform one vertical mapping lap. "
            "If you find a third cylinder, approach it but do not map it. "
            "Then return home."
        ),
        initial_world=MockWorldState(
            cylinders=[],
            completed_steps=[_takeoff(), _search("lawnmower")],
            remaining_steps=[_return_home()],
        ),
        events=[
            # ------------------------------------------------------------------
            # cp0: No cylinders discovered; [return_home] is correct. NOMINAL.
            # ------------------------------------------------------------------
            [],

            # ------------------------------------------------------------------
            # cp1: First cylinder discovered at ~7 m.
            #      Mission: first cylinder → orbit x2.
            #      remaining=[return_home]; no approach(1)/map(1). REPLAN.
            # ------------------------------------------------------------------
            [InjectCylinder(_cyl(1, 7.0)),
             InjectFailure("Cylinder detected at approximately 7 m.")],

            # ------------------------------------------------------------------
            # cp2: Second cylinder detected at ~13 m.
            #      Mission: second cylinder → vertical_map x1.
            #      remaining=[approach(1), map(1,orbit,x2), return_home]; no cyl2. REPLAN.
            # ------------------------------------------------------------------
            [ClearFailure(),
             InjectCylinder(_cyl(2, 13.0)),
             InjectFailure("Cylinder detected at approximately 13 m.")],

            # ------------------------------------------------------------------
            # cp3: Third cylinder detected at ~20 m.
            #      Mission: third cylinder → approach ONLY, no map.
            #      remaining includes cyl1+cyl2 sequences but no cyl3. REPLAN.
            # ------------------------------------------------------------------
            [ClearFailure(),
             InjectCylinder(_cyl(3, 20.0)),
             InjectFailure("Cylinder detected at approximately 20 m.")],

            # ------------------------------------------------------------------
            # cp4: AdvanceSteps(5) advances:
            #        approach(1)+map(1,orbit,x2)+approach(2)+map(2,vmap)+approach(3)
            #      → history. return_home is the only remaining step. NOMINAL.
            # ------------------------------------------------------------------
            [ClearFailure(), AdvanceSteps(5)],
        ],
        expected_nominal=[True, False, False, False, True],
        expected_tails=[
            # cp0: NOMINAL — no reference tail.
            None,
            # cp1: REPLAN — first cylinder → orbit x2.
            [_approach(1), _map_orbit(1, repeat=2), _return_home()],
            # cp2: REPLAN — keep cyl1 sequence; add cyl2 vertical_map before return_home.
            [_approach(1), _map_orbit(1, repeat=2),
             _approach(2), _map_vmap(2), _return_home()],
            # cp3: REPLAN — keep cyl1+cyl2; add approach(3) ONLY (no map(3)).
            [_approach(1), _map_orbit(1, repeat=2),
             _approach(2), _map_vmap(2),
             _approach(3), _return_home()],
            # cp4: NOMINAL — no reference tail.
            None,
        ],
        expected_reasoning=[
            "No cylinders discovered; [return_home] satisfies the mission. NOMINAL.",
            "Cylinder 1 (first) detected at ~7 m. Mission requires orbit x2. "
            "remaining=[return_home] has no approach(1)/map(1). Replanner must insert "
            "approach(1)+map(orbit,x2) before return_home. REPLAN.",
            "Cylinder 2 (second) detected at ~13 m. Mission requires vertical_map x1. "
            "Current tail covers cylinder 1 but not cylinder 2. Replanner must also "
            "insert approach(2)+map(vertical_map) before return_home. REPLAN.",
            "Cylinder 3 (third) detected at ~20 m. Mission requires approach-only — no "
            "map. Replanner must insert approach(3) WITHOUT any map(3) step. REPLAN.",
            "All planned steps (approach/map x1x2 / vmap x1 / approach-only) advanced. "
            "return_home remains. NOMINAL.",
        ],
        note=(
            "C3: Heterogeneous per-cylinder instructions — the mission assigns a DIFFERENT "
            "action to each discovery order: orbit x2 (cyl 1), vertical_map (cyl 2), "
            "approach-only / no map (cyl 3). Cylinders arrive one-at-a-time via "
            "InjectCylinder. Replanner must read the mission text carefully to assign "
            "the correct step type to each cylinder based on discovery order, not ID. "
            "Reference tails enforce mode+repeat correctness at each checkpoint."
        ),
    ),

    # =========================================================================
    # LONG — end-to-end no-failure regression checks
    # =========================================================================

    ChainScenario(
        id="L1",
        complexity="long",
        mission="Search for cylinders and map every one you find, then return.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0), _cyl(3, 15.0)],
            completed_steps=[_takeoff(), _search("lawnmower")],
            remaining_steps=[
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
        ),
        events=[
            [],
            [AdvanceSteps(2)],   # approach(1)+map(1) → history
            [AdvanceSteps(2)],   # approach(2)+map(2) → history
            [AdvanceSteps(2)],   # approach(3)+map(3) → history; return_home remains
        ],
        expected_nominal=[True, True, True, True],
        expected_tails=[None, None, None, None],
        expected_reasoning=[
            "All three cylinders covered with correct approach→map pairs. No failure. NOMINAL.",
            "Cylinder 1 mapped; remaining plan still valid for cylinders 2 and 3. NOMINAL.",
            "Cylinders 1-2 mapped; remaining plan still valid for cylinder 3. NOMINAL.",
            "All three cylinders mapped; return_home is the only remaining step. NOMINAL.",
        ],
        note=(
            "Full three-cylinder mission with NO failures. Every checkpoint should produce "
            "NOMINAL. Primary regression check: validates the replanner does NOT fabricate "
            "changes when the plan is already correct and progressing normally."
        ),
    ),

    ChainScenario(
        id="L2",
        complexity="long",
        mission="Do a thorough search then do 2 vertical mapping laps of each cylinder found.",
        initial_world=MockWorldState(
            cylinders=[_cyl(1, 6.0), _cyl(2, 12.0)],
            completed_steps=[_takeoff(6.0), _search("lawnmower")],
            remaining_steps=[
                _approach(1, sd=5.0), _map_vmap(1, sd=5.0, repeat=2),
                _approach(2, sd=5.0), _map_vmap(2, sd=5.0, repeat=2),
                _return_home(),
            ],
        ),
        events=[
            [],
            [AdvanceSteps(2)],   # approach(1)+map_vmap(1,x2) → history
            [AdvanceSteps(2)],   # approach(2)+map_vmap(2,x2) → history; home remains
        ],
        expected_nominal=[True, True, True],
        expected_tails=[None, None, None],
        expected_reasoning=[
            "Both cylinders planned with approach→vmap(x2). No failure. NOMINAL.",
            "Cylinder 1 mapped (2 vmap laps); remaining plan still valid for cylinder 2. NOMINAL.",
            "Both cylinders mapped (2 vmap laps each); return_home is the only remaining "
            "step. NOMINAL.",
        ],
        note=(
            "Two-cylinder vertical-mapping mission with NO failures. All checkpoints must "
            "produce NOMINAL. Regression check specifically for the vmap mission type: "
            "validates the replanner handles repeat-lap plans without spurious changes."
        ),
    ),
]


# ---------------------------------------------------------------------------
# CHAIN SIMULATION ENGINE
# ---------------------------------------------------------------------------


def _apply_events(world: MockWorldState, events: list) -> MockWorldState:
    w = copy.deepcopy(world)
    for ev in events:
        if isinstance(ev, AdvanceSteps):
            for _ in range(ev.n):
                if w.remaining_steps:
                    w.completed_steps.append(w.remaining_steps.pop(0))
        elif isinstance(ev, MarkCurrentStepFailed):
            if w.remaining_steps:
                w.remaining_steps.pop(0)
            w.failure_context = ev.context
        elif isinstance(ev, InjectFailure):
            w.failure_context = ev.context
        elif isinstance(ev, ClearFailure):
            w.failure_context = None
        elif isinstance(ev, InjectCylinder):
            if ev.cylinder.id not in {c.id for c in w.cylinders}:
                w.cylinders.append(ev.cylinder)
        elif isinstance(ev, UpdateBattery):
            w.battery_pct = ev.pct
    return w


def _count_state_drift(completed: list[dict], tail: list[dict]) -> int:
    """
    Count how many cylinders are re-introduced into the tail that were
    already mapped — without an explicit re-map signal in the failure context.
    """
    mapped = set(_extract_mapped_cylinders(completed).keys())
    if not mapped or not tail:
        return 0
    tail_map_ids = {
        s.get("args", {}).get("cylinder_id")
        for s in tail
        if isinstance(s, dict) and s.get("state") == "map"
        and isinstance(s.get("args", {}).get("cylinder_id"), int)
    }
    return len(mapped & tail_map_ids)


def _classify_decision(nominal: bool, expected_nominal: bool | None) -> str | None:
    """
    Classify the binary NOMINAL/REPLAN decision against ground truth.

    Confusion matrix terminology (REPLAN = positive class):

      True Positive  (TP): Expected REPLAN, replanner said REPLAN
      False Positive (FP): Expected NOMINAL, replanner said REPLAN  ← spurious change
      True Negative  (TN): Expected NOMINAL, replanner said NOMINAL  ← correct quiescence
      False Negative (FN): Expected REPLAN, replanner said NOMINAL   ← missed failure

    Returns one of "TP", "FP", "TN", "FN", or None (unscored checkpoint).

    NOTE: This is the decision-only classification used for confusion matrix stats.
    It does NOT assess whether the tail content was correct. See
    _check_behavior_accuracy() for the composite metric.
    """
    if expected_nominal is None:
        return None
    if not nominal and not expected_nominal:
        return "TP"   # expected REPLAN, said REPLAN
    if nominal and expected_nominal:
        return "TN"   # expected NOMINAL, said NOMINAL
    if not nominal and expected_nominal:
        return "FP"   # expected NOMINAL, said REPLAN  (spurious change)
    # nominal and not expected_nominal
    return "FN"       # expected REPLAN, said NOMINAL  (missed failure)


def _check_behavior_accuracy(
    nominal: bool,
    expected_nominal: bool | None,
    valid: bool,
    tail_match: bool | None,
) -> bool | None:
    """
    Composite behavior correctness for a single checkpoint — the PRIMARY quality metric.

    For a NOMINAL checkpoint (expected_nominal is True):
      Correct iff replanner also said NOMINAL (TN). No tail to compare.

    For a REPLAN checkpoint (expected_nominal is False):
      Correct iff ALL three conditions hold:
        1. Replanner chose REPLAN          (TP — correct decision)
        2. Produced tail is structurally valid (passes validate_plan())
        3. Produced tail matches the reference tail (_tails_match() returned True)

    Returning False for "correct decision but wrong tail" ensures behavior_accuracy
    captures end-to-end replanner quality, not just binary decision accuracy.

    tail_match:
      True  — _tails_match() confirmed the tail matches the reference.
      False — _tails_match() found a discrepancy.
      None  — no reference tail for this checkpoint (unscored content check);
              for REPLAN checkpoints this means structural validity is sufficient
              (content is unchecked), so the condition is treated as True.

    Returns True / False / None (None = unscored checkpoint).
    """
    if expected_nominal is None:
        return None

    if expected_nominal:
        # Expected NOMINAL: correct iff replanner also said NOMINAL.
        return nominal

    # Expected REPLAN:
    if nominal:
        return False          # FN — missed the failure
    if not valid:
        return False          # Right decision but structurally broken tail
    if tail_match is False:
        return False          # Right decision, valid structure, wrong content
    # tail_match is True or None (no reference → structural validity sufficient)
    return True


def run_chain(scenario: ChainScenario, model_key: str) -> dict:
    """
    Simulate one full chain scenario for a given model.

    Per-checkpoint tracking:
      - decision_class      : "TP" / "FP" / "TN" / "FN" (confusion matrix cell)
      - behavior_correct    : bool — composite correctness (decision + validity + tail match)
      - tail_match          : bool | None — did produced tail match reference tail?
      - tail_match_detail   : str — human-readable explanation from _tails_match()

    Aggregate metrics derived from these at the end of the chain.
    """
    world         = copy.deepcopy(scenario.initial_world)
    checkpoints   = []
    total_latency = 0.0
    chain_valid   = True
    reached_home  = False

    replan_cp_indices: list[int] = []
    recovery_gaps:     list[int] = []

    n_expected   = len(scenario.expected_nominal)
    n_ref_tails  = len(scenario.expected_tails)

    for cp_idx, event_list in enumerate(scenario.events):
        world = _apply_events(world, event_list)

        expected_nominal = (
            scenario.expected_nominal[cp_idx]
            if cp_idx < n_expected else None
        )
        expected_tail = (
            scenario.expected_tails[cp_idx]
            if cp_idx < n_ref_tails else None
        )
        expected_reasoning = (
            scenario.expected_reasoning[cp_idx]
            if cp_idx < len(scenario.expected_reasoning) else ""
        )

        # Skip checkpoints where the chain has cleanly converged.
        is_final_home = (
            len(world.remaining_steps) == 1
            and world.remaining_steps[0].get("state") == "return_home"
            and not world.failure_context
        )
        no_steps  = not world.remaining_steps
        quiescent = (is_final_home or no_steps) and not any(
            isinstance(e, (InjectCylinder, InjectFailure, MarkCurrentStepFailed))
            for e in event_list
        )

        if quiescent:
            reached_home = True
            if replan_cp_indices:
                last_replan = replan_cp_indices[-1]
                recovery_gaps.append(cp_idx - last_replan)
                replan_cp_indices.clear()

            checkpoints.append({
                "checkpoint":         cp_idx,
                "skipped":            True,
                "reason":             "Chain converged to return_home",
                "expected_nominal":   expected_nominal,
                "expected_reasoning": expected_reasoning,
            })
            continue

        result = run_replan(
            mission_intent=scenario.mission,
            world=world,
            model_key=model_key,
        )
        total_latency += result["latency_s"]

        # ---------------------------------------------------------------
        # Drift and plan-delta (only when a new tail was produced).
        # ---------------------------------------------------------------
        drift      = 0
        plan_delta = 0
        if not result["nominal"] and result["tail"]:
            re_map_signal = (
                world.failure_context is not None
                and any(kw in (world.failure_context or "").lower()
                        for kw in ["unusable", "corrupted", "corrupt",
                                   "0 valid", "0 completed", "not been mapped",
                                   "no mapping", "has not been mapped"])
            )
            raw_drift  = _count_state_drift(world.completed_steps, result["tail"])
            drift      = 0 if re_map_signal else raw_drift
            plan_delta = len(result["tail"]) - len(world.remaining_steps)
            world.remaining_steps = copy.deepcopy(result["tail"])

        if not result["valid"]:
            chain_valid = False

        # ---------------------------------------------------------------
        # Confusion matrix classification (decision only).
        # ---------------------------------------------------------------
        decision_class = _classify_decision(result["nominal"], expected_nominal)

        # ---------------------------------------------------------------
        # Tail matching against reference tail.
        # Only runs for REPLAN checkpoints that have a reference tail.
        # ---------------------------------------------------------------
        if not result["nominal"] and expected_tail is not None:
            tail_match, tail_match_detail = _tails_match(
                result.get("tail"), expected_tail
            )
        elif result["nominal"] and expected_tail is None:
            # Correct NOMINAL with no reference tail — content check N/A.
            tail_match, tail_match_detail = None, "NOMINAL — no tail to compare."
        elif result["nominal"] and expected_tail is not None:
            # FN — said NOMINAL but should have replanned (has a reference tail).
            tail_match       = False
            tail_match_detail = "Replanner said NOMINAL; reference tail exists."
        else:
            # REPLAN with no reference tail — content check skipped.
            tail_match, tail_match_detail = None, "No reference tail — content check skipped."

        # ---------------------------------------------------------------
        # Composite behavior correctness.
        # ---------------------------------------------------------------
        behavior_correct = _check_behavior_accuracy(
            nominal=result["nominal"],
            expected_nominal=expected_nominal,
            valid=result["valid"],
            tail_match=tail_match,
        )

        # A tail mismatch or structural failure also counts as chain invalid.
        if tail_match is False or not result["valid"]:
            chain_valid = False

        # ---------------------------------------------------------------
        # Recovery tracking (based on valid REPLAN decisions).
        # ---------------------------------------------------------------
        if not result["nominal"] and result["valid"]:
            replan_cp_indices.append(cp_idx)
        elif result["nominal"] and replan_cp_indices:
            last_replan = replan_cp_indices[-1]
            recovery_gaps.append(cp_idx - last_replan)
            replan_cp_indices.clear()

        checkpoints.append({
            "checkpoint":          cp_idx,
            "skipped":             False,
            "events":              [type(e).__name__ for e in event_list],
            "failure_context":     world.failure_context,
            "nominal":             result["nominal"],
            "valid":               result["valid"],
            "drift":               drift,
            "plan_delta":          plan_delta,
            "attempts":            result["attempts"],
            "latency_s":           result["latency_s"],
            "reason":              result.get("reason"),
            "tail_states":         [s.get("state") for s in (result.get("tail") or [])],
            # Reference tail for traceability.
            "reference_tail_states": (
                [s.get("state") for s in expected_tail] if expected_tail else None
            ),
            "expected_nominal":    expected_nominal,
            "expected_reasoning":  expected_reasoning,
            # Confusion matrix cell.
            "decision_class":      decision_class,
            # Tail content match.
            "tail_match":          tail_match,
            "tail_match_detail":   tail_match_detail,
            # Composite behavior accuracy.
            "behavior_correct":    behavior_correct,
            "errors":              result.get("errors", []),
        })

    final_states = [s.get("state") for s in world.remaining_steps]
    reached_home = reached_home or (
        len(world.remaining_steps) <= 1
        and (not world.remaining_steps
             or world.remaining_steps[0].get("state") == "return_home")
    )

    active_cps = [c for c in checkpoints if not c.get("skipped")]
    n_active   = len(active_cps)

    # ---------------------------------------------------------------
    # Confusion matrix totals (decision-only, REPLAN = positive class).
    #   TP: expected REPLAN, said REPLAN
    #   FP: expected NOMINAL, said REPLAN  (spurious change)
    #   TN: expected NOMINAL, said NOMINAL  (correct quiescence)
    #   FN: expected REPLAN, said NOMINAL   (missed failure)
    # ---------------------------------------------------------------
    cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for cp in active_cps:
        dc = cp.get("decision_class")
        if dc in cm:
            cm[dc] += 1

    # Behavior accuracy (composite).
    behavior_labeled = [c for c in active_cps if c.get("behavior_correct") is not None]
    behavior_correct_count = sum(1 for c in behavior_labeled if c["behavior_correct"])

    # Tail-match summary (only REPLAN checkpoints with reference tails).
    tail_check_cps    = [c for c in active_cps if c.get("tail_match") is not None]
    n_tail_checks     = len(tail_check_cps)
    n_tail_match_ok   = sum(1 for c in tail_check_cps if c.get("tail_match"))

    nominal_total = sum(1 for cp in active_cps if cp.get("nominal"))

    return {
        "scenario_id":             scenario.id,
        "complexity":              scenario.complexity,
        "model":                   model_key,
        "mission":                 scenario.mission,
        "chain_valid":             chain_valid,
        "reached_home":            reached_home,
        "total_latency":           round(total_latency, 2),
        "n_checkpoints":           n_active,
        "nominal_calls":           nominal_total,
        # Composite behavior accuracy (PRIMARY metric).
        "behavior_correct_count":  behavior_correct_count,
        "behavior_total":          len(behavior_labeled),
        # Confusion matrix (decision-only, REPLAN = positive class).
        "confusion_matrix":        cm,
        # Tail content match stats.
        "n_tail_checks":           n_tail_checks,
        "n_tail_match_ok":         n_tail_match_ok,
        # Recovery sequence data.
        "recovery_gaps":           recovery_gaps,
        "final_remaining_states":  final_states,
        "checkpoints":             checkpoints,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def run_experiment(model_keys: list[str], output_dir: str = "exp3_output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list] = {mk: [] for mk in model_keys}

    print("\n" + "=" * 72)
    print("EXPERIMENT 3: MULTI-STEP REPLANNING CHAINS (single failure/checkpoint)")
    print("=" * 72)
    print("Complexity tiers:")
    print("  simple  — 1 cylinder, 1 failure, basic gap detection")
    print("  medium  — 2 cylinders, 2-3 sequential single failures")
    print("  complex — varied failures: late discovery / abort-policy / heterogeneous")
    print("  long    — no failures; validates replanner leaves good plans alone\n")

    complexities = ["simple", "medium", "complex", "long"]

    for scenario in CHAINS:
        print(f"\n--- [{scenario.id}] {scenario.complexity.upper()}: {scenario.note[:70]}...")
        for model_key in model_keys:
            print(f"  Model: {model_key}")
            chain_result = run_chain(scenario, model_key)
            all_results[model_key].append(chain_result)

            n_cp   = chain_result["n_checkpoints"]
            b_cor  = chain_result["behavior_correct_count"]
            b_tot  = chain_result["behavior_total"]
            drifts = sum(c.get("drift", 0) for c in chain_result["checkpoints"]
                         if not c.get("skipped"))
            cm     = chain_result["confusion_matrix"]
            n_tc   = chain_result["n_tail_checks"]
            n_tok  = chain_result["n_tail_match_ok"]

            print(
                f"    chain_valid={chain_result['chain_valid']}  "
                f"reached_home={chain_result['reached_home']}  "
                f"checkpoints={n_cp}  "
                f"behavior_correct={b_cor}/{b_tot}  "
                f"drift={drifts}  "
                + (f"tail_match={n_tok}/{n_tc}  " if n_tc > 0 else "")
                + f"latency={chain_result['total_latency']:.1f}s"
            )
            print(
                f"      confusion matrix — "
                f"TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}"
            )
            for cp in chain_result["checkpoints"]:
                if cp.get("skipped"):
                    print(f"      cp{cp['checkpoint']}: skipped ({cp['reason']})")
                    continue

                beh_mark  = ("✓" if cp["behavior_correct"] else "✗") if cp.get("behavior_correct") is not None else "?"
                tail_mark = ""
                if cp.get("tail_match") is not None:
                    tail_mark = f"  tail:{'✓' if cp['tail_match'] else '✗'}"
                dc_str = f"  [{cp['decision_class']}]" if cp.get("decision_class") else ""

                print(
                    f"      cp{cp['checkpoint']}: "
                    f"{'NOMINAL' if cp['nominal'] else ('VALID_REPLAN' if cp['valid'] else 'INVALID')}  "
                    f"drift={cp['drift']}  delta={cp['plan_delta']:+d}  "
                    f"events={cp['events']}{dc_str}  [behav:{beh_mark}]{tail_mark}"
                )
                if cp.get("tail_match") is False:
                    print(f"        tail_mismatch: {cp['tail_match_detail']}")
                if cp.get("reference_tail_states") is not None:
                    print(f"        reference: {cp['reference_tail_states']}")
                    print(f"        produced:  {cp['tail_states']}")
                if cp.get("expected_reasoning"):
                    print(f"        expected: {cp['expected_reasoning']}")

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    def _agg(recs: list[dict]) -> dict:
        """Compute aggregate metrics over a set of chain results."""
        if not recs:
            return {}
        n       = len(recs)
        all_cps = [c for r in recs for c in r["checkpoints"] if not c.get("skipped")]
        n_cps   = len(all_cps)
        if n_cps == 0:
            return {"n_chains": n, "note": "no active checkpoints"}

        chain_success = sum(1 for r in recs if r["chain_valid"] and r["reached_home"])
        valid_calls   = sum(1 for c in all_cps if c.get("valid"))
        drift_events  = sum(c.get("drift", 0) for c in all_cps)
        plan_deltas   = [c["plan_delta"] for c in all_cps if not c.get("nominal")]

        # ------------------------------------------------------------------
        # Composite behavior accuracy (PRIMARY metric).
        # Correct iff: right NOMINAL/REPLAN decision AND (for REPLAN) valid
        # tail AND tail matches reference.
        # ------------------------------------------------------------------
        labeled_behavior = [
            c for c in all_cps if c.get("behavior_correct") is not None
        ]
        n_labeled_behavior = len(labeled_behavior)
        behavior_correct_n = sum(1 for c in labeled_behavior if c["behavior_correct"])

        # ------------------------------------------------------------------
        # Confusion matrix (REPLAN = positive class).
        #   TP: expected REPLAN, said REPLAN
        #   FP: expected NOMINAL, said REPLAN  (spurious change)
        #   TN: expected NOMINAL, said NOMINAL
        #   FN: expected REPLAN, said NOMINAL   (missed failure)
        # ------------------------------------------------------------------
        cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for c in all_cps:
            dc = c.get("decision_class")
            if dc in cm:
                cm[dc] += 1

        n_labeled_dec  = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        decision_correct = cm["TP"] + cm["TN"]

        # Derived rates.
        n_replan_expected  = cm["TP"] + cm["FN"]   # all checkpoints that should REPLAN
        n_nominal_expected = cm["TN"] + cm["FP"]   # all checkpoints that should NOMINAL

        # Precision, Recall, F1 (REPLAN = positive class).
        precision = cm["TP"] / max(cm["TP"] + cm["FP"], 1)
        recall    = cm["TP"] / max(cm["TP"] + cm["FN"], 1)   # = TPR / sensitivity
        f1        = (2 * precision * recall / max(precision + recall, 1e-9))

        # False positive rate (FP / all expected NOMINAL).
        fpr = cm["FP"] / max(n_nominal_expected, 1)
        # False negative rate (FN / all expected REPLAN).
        fnr = cm["FN"] / max(n_replan_expected, 1)

        # Tail match stats.
        tail_check_cps  = [c for c in all_cps if c.get("tail_match") is not None]
        n_tail_checks   = len(tail_check_cps)
        n_tail_match_ok = sum(1 for c in tail_check_cps if c.get("tail_match"))

        # Recovery metrics.
        all_gaps        = [g for r in recs for g in r.get("recovery_gaps", [])]
        n_valid_replans = sum(
            1 for c in all_cps if not c.get("nominal") and c.get("valid")
        )

        return {
            "n_chains": n,
            # Primary outcome.
            "chain_success_pct":       round(100 * chain_success / n,                   1),
            # Structural validity.
            "per_step_valid_pct":      round(100 * valid_calls / n_cps,                 1),
            # PRIMARY QUALITY METRIC: composite behavior accuracy.
            # = % of labeled checkpoints where decision was correct AND
            #   (for REPLAN) produced tail matched reference tail.
            "behavior_accuracy_pct":   round(
                100 * behavior_correct_n / max(n_labeled_behavior, 1),                  1),
            # Decision-only accuracy (binary NOMINAL/REPLAN) — for comparison.
            # Compare with behavior_accuracy_pct: gap = tail-content error rate.
            "decision_accuracy_pct":   round(
                100 * decision_correct / max(n_labeled_dec, 1),                         1),
            # Confusion matrix raw counts (REPLAN = positive class).
            "cm_TP":                   cm["TP"],
            "cm_FP":                   cm["FP"],
            "cm_TN":                   cm["TN"],
            "cm_FN":                   cm["FN"],
            # Derived confusion matrix rates.
            "true_positive_rate":      round(100 * recall,    1),   # sensitivity / recall
            "true_negative_rate":      round(
                100 * cm["TN"] / max(n_nominal_expected, 1),                            1),
            "false_positive_rate":     round(100 * fpr,       1),   # spurious replans
            "false_negative_rate":     round(100 * fnr,       1),   # missed failures
            "precision":               round(100 * precision,  1),
            "f1_score":                round(100 * f1,         1),
            # Per-class correct rates.
            "correct_nominal_pct":     round(
                100 * cm["TN"] / max(n_nominal_expected, 1),                            1),
            "correct_replan_pct":      round(
                100 * cm["TP"] / max(n_replan_expected, 1),                             1),
            # Tail match stats.
            "tail_match_pct":          (
                round(100 * n_tail_match_ok / n_tail_checks, 1) if n_tail_checks > 0 else None
            ),
            "n_tail_checks":           n_tail_checks,
            # Plan quality.
            "state_drift_events":      drift_events,
            "avg_plan_length_delta":   round(sum(plan_deltas) / max(len(plan_deltas), 1), 2),
            # Nominal/replan balance.
            "nominal_rate_pct":        round(
                100 * sum(1 for c in all_cps if c.get("nominal")) / n_cps,             1),
            # Recovery.
            "recovery_rate_pct":       round(
                100 * len(all_gaps) / max(n_valid_replans, 1),                          1),
            "avg_steps_to_recover":    round(
                sum(all_gaps) / max(len(all_gaps), 1),                                  2),
            # Latency.
            "avg_checkpoints":         round(sum(r["n_checkpoints"] for r in recs) / n, 1),
            "avg_chain_latency_s":     round(sum(r["total_latency"] for r in recs) / n, 2),
        }

    summary: dict[str, dict] = {}
    for mk in model_keys:
        records  = all_results[mk]
        by_cx    = {cx: _agg([r for r in records if r["complexity"] == cx])
                    for cx in complexities}
        summary[mk] = {"overall": _agg(records), "by_complexity": by_cx}

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 3 SUMMARY — Table 1: Primary Outcomes (Overall)")
    print()
    print("  Confusion matrix uses REPLAN as the positive class:")
    print("    TP = expected REPLAN, said REPLAN")
    print("    FP = expected NOMINAL, said REPLAN   (spurious change — conservative error)")
    print("    TN = expected NOMINAL, said NOMINAL  (correct quiescence)")
    print("    FN = expected REPLAN, said NOMINAL   (missed failure — dangerous error)")
    print()
    print("  behav%   = behavior_accuracy_pct [PRIMARY] — % of labeled checkpoints where")
    print("             the replanner made the correct NOMINAL/REPLAN decision AND")
    print("             (for REPLAN checkpoints) the produced tail matched the reference")
    print("             tail exactly. This is the single best measure of end-to-end quality.")
    print()
    print("  dec%     = decision_accuracy_pct — binary NOMINAL/REPLAN only (TP+TN rate).")
    print("             Compare with behav%: gap = tail-content error contribution.")
    print()
    print("  tail%    = % of REPLAN checkpoints where produced tail matched reference tail.")
    print("             Covers mode, repeat, cylinder ordering, and step types.")
    print()
    print("  FPR%     = false_positive_rate  (FP / expected NOMINAL) — spurious replans.")
    print("  FNR%     = false_negative_rate  (FN / expected REPLAN)  — missed failures.")
    print("  chain%   = % of chains that were fully valid and reached return_home.")
    print("=" * 100)
    cols_p = ["chain%", "behav%", "dec%", "tail%", "FPR%", "FNR%", "F1%", "recover%", "lat(s)"]
    print(f"{'Model':<22} " + "  ".join(f"{c:>9}" for c in cols_p))
    print("-" * 110)
    for mk, s in summary.items():
        o  = s["overall"]
        tm = o.get("tail_match_pct")
        print(
            f"{mk:<22}  "
            f"{o.get('chain_success_pct',     0):>9.1f}  "
            f"{o.get('behavior_accuracy_pct', 0):>9.1f}  "
            f"{o.get('decision_accuracy_pct', 0):>9.1f}  "
            f"{'N/A' if tm is None else f'{tm:.1f}':>9}  "
            f"{o.get('false_positive_rate',   0):>9.1f}  "
            f"{o.get('false_negative_rate',   0):>9.1f}  "
            f"{o.get('f1_score',              0):>9.1f}  "
            f"{o.get('recovery_rate_pct',     0):>9.1f}  "
            f"{o.get('avg_chain_latency_s',   0):>8.2f}"
        )

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 3 SUMMARY — Table 2: Confusion Matrix (Overall, REPLAN = positive class)")
    print()
    print("  TP = True  Positive (correct REPLAN)  FP = False Positive (spurious REPLAN)")
    print("  TN = True  Negative (correct NOMINAL)  FN = False Negative (missed failure)")
    print()
    print("  TPR% = True  Positive Rate = recall = TP / (TP+FN)  — sensitivity")
    print("  TNR% = True  Negative Rate          = TN / (TN+FP)  — specificity")
    print("  FPR% = False Positive Rate           = FP / (TN+FP)")
    print("  FNR% = False Negative Rate           = FN / (TP+FN)")
    print("  Prec = Precision                     = TP / (TP+FP)")
    print("  F1   = 2·Prec·Recall / (Prec+Recall)")
    print("  drift= state drift events (already-mapped cylinders re-added to tail)")
    print("  Δplan= avg tail-length change per non-NOMINAL call")
    print("=" * 100)
    cols_a = ["TP", "FP", "TN", "FN", "TPR%", "TNR%", "FPR%", "FNR%", "Prec%", "F1%", "drift", "Δplan"]
    print(f"{'Model':<22} " + "  ".join(f"{c:>6}" for c in cols_a))
    print("-" * 120)
    for mk, s in summary.items():
        o = s["overall"]
        print(
            f"{mk:<22}  "
            f"{o.get('cm_TP',                0):>6}  "
            f"{o.get('cm_FP',                0):>6}  "
            f"{o.get('cm_TN',                0):>6}  "
            f"{o.get('cm_FN',                0):>6}  "
            f"{o.get('true_positive_rate',   0):>6.1f}  "
            f"{o.get('true_negative_rate',   0):>6.1f}  "
            f"{o.get('false_positive_rate',  0):>6.1f}  "
            f"{o.get('false_negative_rate',  0):>6.1f}  "
            f"{o.get('precision',            0):>6.1f}  "
            f"{o.get('f1_score',             0):>6.1f}  "
            f"{o.get('state_drift_events',   0):>6}  "
            f"{o.get('avg_plan_length_delta',0):>6.2f}"
        )

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 3 SUMMARY — Table 3: Mission-Type Breakdown")
    print("  behav% = composite (decision + tail match), dec% = decision only.")
    print("  Gap between dec% and behav% isolates tail-content error rate per tier.")
    print("  complex tier includes abort-policy (C2) and heterogeneous (C3) scenarios.")
    print("=" * 100)
    for cx in complexities:
        print(f"\n  [{cx.upper()}]")
        print(f"  {'Model':<22} " + "  ".join(f"{c:>9}" for c in
              ["chain%", "behav%", "dec%", "tail%", "FPR%", "FNR%", "drift", "lat(s)"]))
        print("  " + "-" * 110)
        for mk, s in summary.items():
            b = s["by_complexity"].get(cx, {})
            if not b or "chain_success_pct" not in b:
                print(f"  {mk:<22}  (no records)")
                continue
            tm = b.get("tail_match_pct")
            print(
                f"  {mk:<22}  "
                f"{b['chain_success_pct']:>9.1f}  "
                f"{b['behavior_accuracy_pct']:>9.1f}  "
                f"{b['decision_accuracy_pct']:>9.1f}  "
                f"{'N/A' if tm is None else f'{tm:.1f}':>9}  "
                f"{b['false_positive_rate']:>9.1f}  "
                f"{b['false_negative_rate']:>9.1f}  "
                f"{b['state_drift_events']:>9}  "
                f"{b['avg_chain_latency_s']:>8.2f}"
            )

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 3 SUMMARY — Table 4: Per-Scenario Outcome Matrix")
    print("  behav  = behavior_correct/total (decision + tail match)")
    print("  tail   = tail_match_ok/tail_checks (REPLAN checkpoints only)")
    print("  TP/FP/TN/FN = confusion matrix counts for this chain")
    print("=" * 100)
    for mk in model_keys:
        print(f"\n  Model: {mk}")
        print(f"  {'ID':<6} {'cx':<9} {'valid':<7} {'home':<7} "
              f"{'behav':>7} "
              f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
              f"{'tail':>8} {'drift':>6} {'lat':>7}")
        print("  " + "-" * 90)
        for r in all_results[mk]:
            cm     = r["confusion_matrix"]
            ntc    = r["n_tail_checks"]
            ntok   = r["n_tail_match_ok"]
            bc     = r["behavior_correct_count"]
            bt     = r["behavior_total"]
            tail_s = f"{ntok}/{ntc}" if ntc > 0 else "  N/A"
            beh_s  = f"{bc}/{bt}"
            drift  = sum(c.get("drift", 0) for c in r["checkpoints"] if not c.get("skipped"))
            print(
                f"  {r['scenario_id']:<6} {r['complexity']:<9} "
                f"{'YES' if r['chain_valid'] else 'NO':<7} "
                f"{'YES' if r['reached_home'] else 'NO':<7} "
                f"{beh_s:>7} "
                f"{cm['TP']:>4} "
                f"{cm['FP']:>4} "
                f"{cm['TN']:>4} "
                f"{cm['FN']:>4} "
                f"{tail_s:>8} "
                f"{drift:>6} "
                f"{r['total_latency']:>6.1f}s"
            )

    save_json(f"{output_dir}/exp3_results_raw.json", all_results)
    save_json(f"{output_dir}/exp3_summary.json", summary)
    print(f"\n  Saved → {output_dir}/")
    print("\nDone.\n")
    return all_results, summary


if __name__ == "__main__":
    MODELS_TO_TEST = ["gemini-flash-2.5", "qwen-235b", "llama-3.1-8b"]
    run_experiment(MODELS_TO_TEST, output_dir="exp3_output")