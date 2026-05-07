#!/usr/bin/env python3
"""
exp2_replanning_under_failure.py
=================================
Experiment 2: Replanning Under Failure  (main contribution)

Tests the LLM replanner's ability to produce correct, valid plan adaptations
when given realistic mid-mission world states and failure events.

Design principle
----------------
failure_context describes what HAPPENED IN THE WORLD — raw sensor events,
system telemetry, and perception reports. It never describes what the
replanner should do in response. The reasoning about the correct recovery
is the LLM's contribution and is what we are evaluating.

Three sources of replanning signal, in increasing subtlety:
  1. Explicit failure event in failure_context
       ("perception lost track of cylinder 2")
  2. Implicit mismatch between world state and remaining plan
       (cylinder 3 in manifest, not in plan — no context needed)
  3. Pure completion-history reasoning
       (mission said 2 laps, history shows 1 done, remaining plan missing it)

Each scenario specifies:
  - A mission intent (user's original request)
  - completed_steps: execution history (fixed, cannot be changed)
  - remaining_steps: what the original planner scheduled next
  - cylinders: live world model at this point in the mission
  - failure_context: raw event description, or None
  - expected_nominal: True if the correct response is NOMINAL (no change)
  - expected_tail: the exact correct tail if expected_nominal is False

Evaluation framework mirrors Experiments 3 and 4:
  - Tail matching uses _tails_match(), comparing step-by-step on
    "state", "args" (cylinder_id, mode, standoff_distance, min_altitude_m),
    and "repeat".
  - A confusion matrix is accumulated with REPLAN as the positive class:
      TP: expected REPLAN, replanner said REPLAN AND tail matched reference
      FP: expected NOMINAL, replanner said REPLAN   (spurious change)
      TN: expected NOMINAL, replanner said NOMINAL  (correct quiescence)
      FN: expected REPLAN, replanner said NOMINAL   (missed failure)
  - behavior_accuracy_pct [PRIMARY] requires:
      For expected NOMINAL: replanner said NOMINAL              (TN)
      For expected REPLAN:  replanner said REPLAN AND tail is   (TP + correct tail)
                            structurally valid AND matches reference
  - mission_success_pct [exp1-parallel] is identical in computation to
      behavior_accuracy_pct but named to make the cross-experiment comparison
      explicit: % of ALL missions (NOMINAL + REPLAN) where the outcome was
      fully correct. Kept as a separate field so the two metrics can diverge
      independently if partial-credit logic is added later.
  - decision_accuracy_pct covers the binary NOMINAL/REPLAN choice only.
    Gap between decision_accuracy and behavior_accuracy isolates tail-content errors.

Scenarios (12 total):
  Group A - Target interaction failures   (5)  explicit sensor/perception events
  Group B - World state reasoning         (4)  no failure context, pure manifest diff
  Group C - Battery / resource telemetry  (3)  raw telemetry only

Output:
  exp2_results_raw.json
  exp2_summary.json
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from experiment_utils import (
    MODELS, run_replan, MockCylinder, MockWorldState, make_world,
    fmt_plan, save_json, validate_plan,
)

# ---------------------------------------------------------------------------
# TAIL MATCHING
# ---------------------------------------------------------------------------

_ARGS_KEYS = ("cylinder_id", "mode", "standoff_distance", "min_altitude_m")


def _steps_match(produced: dict, reference: dict) -> bool:
    """
    Return True if a produced step matches a reference step.

    Matching rules:
      - "state" must be identical.
      - "repeat" must be identical (defaults to 1 if absent).
      - For each args key present in the reference step, the produced step
        must carry the same value. Extra keys in the produced step that are
        absent from the reference are silently allowed.
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
            continue
        ref_val  = ref_args[key]
        prod_val = prod_args.get(key)
        if isinstance(ref_val, (int, float)) and isinstance(prod_val, (int, float)):
            if abs(float(ref_val) - float(prod_val)) > 1e-6:
                return False
        else:
            if ref_val != prod_val:
                return False
    return True


def _tails_match(
    produced: list[dict] | None,
    reference: list[dict] | None,
) -> tuple[bool, str]:
    """
    Compare a produced tail against the reference tail.

    Returns (match: bool, explanation: str).

    Special cases:
      - reference is None  → scenario is unscored (True, note).
      - produced is None (NOMINAL) and reference is not None → always False.
    """
    if reference is None:
        return True, "No reference tail — scenario is unscored."
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
            mismatches.append(f"  step[{i}]: produced {p!r} ≠ reference {r!r}")
    if mismatches:
        return False, "Step mismatch(es):\n" + "\n".join(mismatches)
    return True, "Tail matches reference exactly."


def _classify_decision(nominal: bool, expected_nominal: bool | None) -> str | None:
    """
    Classify the binary NOMINAL/REPLAN decision against ground truth.

    Positive class = REPLAN.
      TP: expected REPLAN, said REPLAN
      FP: expected NOMINAL, said REPLAN  (spurious change)
      TN: expected NOMINAL, said NOMINAL (correct quiescence)
      FN: expected REPLAN, said NOMINAL  (missed failure)

    Returns one of "TP", "FP", "TN", "FN", or None (unscored).
    """
    if expected_nominal is None:
        return None
    if not nominal and not expected_nominal:
        return "TP"
    if nominal and expected_nominal:
        return "TN"
    if not nominal and expected_nominal:
        return "FP"
    return "FN"


def _check_behavior_accuracy(
    nominal: bool,
    expected_nominal: bool | None,
    valid: bool,
    tail_match: bool | None,
) -> bool | None:
    """
    Composite behavior correctness for a single scenario.

    For a NOMINAL scenario (expected_nominal True):
      Correct iff replanner also said NOMINAL (TN).

    For a REPLAN scenario (expected_nominal False):
      Correct iff ALL three conditions hold:
        1. Replanner chose REPLAN
        2. Produced tail is structurally valid
        3. Produced tail matches the reference tail

    tail_match = None means no reference tail; structural validity suffices.

    Returns True / False / None (None = unscored scenario).
    """
    if expected_nominal is None:
        return None
    if expected_nominal:
        return nominal
    if nominal:
        return False
    if not valid:
        return False
    if tail_match is False:
        return False
    return True


def _check_mission_success(
    nominal: bool,
    expected_nominal: bool | None,
    valid: bool,
    tail_match: bool | None,
) -> bool | None:
    """
    Mission success: % of ALL missions where the outcome was fully correct.

    Mirrors experiment 1's perfect_tail_pct — denominator is all missions,
    not just those where the replanner chose REPLAN.

    For NOMINAL scenarios:
      Correct iff replanner said NOMINAL.

    For REPLAN scenarios:
      Correct iff replanner said REPLAN AND the produced tail is structurally
      valid AND every step exactly matches the reference tail.

    Returns True / False / None (None = unscored scenario).
    """
    return _check_behavior_accuracy(nominal, expected_nominal, valid, tail_match)


def steps_preserved_pct(completed: list[dict], tail: list[dict] | None) -> float | None:
    """% of already-mapped cylinders that do NOT reappear in the replanned tail."""
    if tail is None:
        return None
    from experiment_utils import _extract_mapped_cylinders
    done_ids = set(_extract_mapped_cylinders(completed).keys())
    if not done_ids:
        return 100.0
    tail_map_ids = {
        s.get("args", {}).get("cylinder_id")
        for s in tail
        if isinstance(s, dict) and s.get("state") == "map"
        and isinstance(s.get("args", {}).get("cylinder_id"), int)
    }
    preserved = done_ids - tail_map_ids
    return 100.0 * len(preserved) / len(done_ids)


# ---------------------------------------------------------------------------
# SCENARIO DEFINITION
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id:               str
    group:            str
    mission:          str
    world:            MockWorldState
    expected_nominal: bool
    # expected_tail:
    #   None       — expected_nominal is True (NOMINAL scenarios have no tail)
    #   list[dict] — exact correct tail the replanner must produce when
    #                expected_nominal is False. Compared via _tails_match().
    # Invariant: if expected_nominal is True, expected_tail must be None.
    expected_tail:    list[dict] | None
    note:             str
    signal_type:      str = "explicit"   # "explicit" | "implicit" | "history"
    rationale:        str = ""


def _step(state, args=None, repeat=1):
    return {"state": state, "args": args or {}, "repeat": repeat}

def _approach(cid, sd=5.0):
    return _step("approach", {"cylinder_id": cid, "standoff_distance": sd})

def _map_orbit(cid, sd=5.0, repeat=1):
    return _step("map", {"cylinder_id": cid, "standoff_distance": sd, "mode": "orbit"}, repeat)

def _map_vmap(cid, sd=5.0, repeat=1):
    return _step("map", {"cylinder_id": cid, "standoff_distance": sd,
                         "mode": "vertical_map", "min_altitude_m": 2.0}, repeat)

def _return_home():
    return _step("return_home")


SCENARIOS: list[Scenario] = [

    # =========================================================================
    # GROUP A – Explicit sensor / perception failure events
    # failure_context = raw perception report only; no instructions embedded
    # =========================================================================

    Scenario(
        id="A1",
        group="A",
        signal_type="explicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1),
            ],
            remaining_steps=[_approach(2), _map_orbit(2), _return_home()],
            failure_context=(
                "Perception lost track of cylinder 2 during approach. "
                "No confirmed detection in the last 10 seconds."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_approach(2), _map_orbit(2), _return_home()],
        note="Target lost mid-approach. Replanner should re-confirm approach before mapping.",
        rationale=(
            "Target lost during approach — standoff geometry is unconfirmed. "
            "Replanner must re-insert approach(2) before map(2) to re-establish "
            "visual contact. The remaining plan already has approach(2)+map(2) "
            "so the correct tail is the same structure — the key test is whether "
            "the replanner recognises that the current approach is invalid and "
            "must be restarted."
        ),
    ),

    Scenario(
        id="A2",
        group="A",
        signal_type="history",
        mission="Find cylinders and do 2 vertical mapping laps of each one you find.",
        world=MockWorldState(
            cylinders=[MockCylinder(id=1, world_x=5.0, world_y=0.0, depth_m=4.0)],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1, sd=5.0),
                _map_vmap(1, sd=5.0, repeat=1),
            ],
            remaining_steps=[_return_home()],
            failure_context=None,
        ),
        expected_nominal=False,
        expected_tail=[_approach(1, sd=5.0), _map_vmap(1, sd=5.0, repeat=1), _return_home()],
        note=(
            "History shows only 1 of 2 requested laps completed. "
            "Remaining plan skips to return_home. "
            "Replanner must catch this from the completion record alone — no hint given."
        ),
        rationale=(
            "History shows only 1 of 2 requested laps completed. Remaining plan "
            "skips to return_home. Replanner must catch this from the completion "
            "record alone — no hint given. Correct tail: re-approach + 1 remaining "
            "lap + return_home."
        ),
    ),

    Scenario(
        id="A3",
        group="A",
        signal_type="explicit",
        mission="Find cylinders and orbit each one twice, then come home.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0, world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1, repeat=2),
                _approach(2),
            ],
            remaining_steps=[_map_orbit(2, repeat=2), _return_home()],
            failure_context=(
                "Setpoint timeout while orbiting cylinder 2."
                "The drone is holding position at the last commanded waypoint."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_approach(2), _map_orbit(2, repeat=2), _return_home()],
        note=(
            "Orbit aborted mid-way through. Remaining plan still shows the "
            "full 2-orbit step. Replanner should re-approach then retry full orbit."
        ),
        rationale=(
            "Orbit aborted mid-way. Drone position is uncertain. Replanner must "
            "re-insert approach(2) before the 2-orbit map step to re-establish "
            "standoff geometry before retrying the full 2-orbit sequence."
        ),
    ),

    Scenario(
        id="A4",
        group="A",
        signal_type="explicit",
        mission="Search for cylinders and do a vertical map of each one.",
        world=MockWorldState(
            cylinders=[MockCylinder(id=1, world_x=6.0, world_y=0.0, depth_m=5.0)],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "lawnmower"}),
                _approach(1, sd=5.0),
            ],
            remaining_steps=[_map_vmap(1, sd=5.0), _return_home()],
            failure_context=(
                "Vertical map skill failed to initialize: cylinder 1 centroid "
                "is outside the camera frame. Drone yaw may have drifted."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_approach(1, sd=5.0), _map_vmap(1, sd=5.0), _return_home()],
        note=(
            "Vertical map cannot start because the target is out of frame. "
            "Replanner should re-approach to re-center."
        ),
        rationale=(
            "Vertical map cannot start because the target is out of frame — the "
            "approach left the drone with incorrect heading. Replanner must insert "
            "a fresh approach(1) before the existing map(1) step."
        ),
    ),

    Scenario(
        id="A5",
        group="A",
        signal_type="explicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
            ],
            remaining_steps=[_return_home()],
            failure_context=None,
        ),
        expected_nominal=True,
        expected_tail=None,
        note=(
            "All discovered cylinders mapped. Only return_home remains. "
            "Correct answer is NOMINAL. Tests false-replan rate."
        ),
        rationale=(
            "All discovered cylinders mapped. Only return_home remains. "
            "Correct answer is NOMINAL. Tests false-replan rate."
        ),
    ),

    # =========================================================================
    # GROUP B – Implicit world-state reasoning (no failure_context)
    # The replanner must compare the cylinder manifest against the remaining
    # plan entirely on its own, with no textual hints.
    # =========================================================================

    Scenario(
        id="B1",
        group="B",
        signal_type="implicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
            ],
            remaining_steps=[
                _approach("all"), _map_orbit("all"),
                _return_home(),
            ],
            failure_context=None,
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        note=(
            "Search completed. Cylinder manifest is empty. Remaining plan still "
            "has approach+map steps. Replanner must infer these are now pointless "
            "from the empty manifest alone."
        ),
        rationale=(
            "Search completed with an empty manifest. The approach+map steps "
            "target 'all' cylinders but none were found — these steps are now "
            "pointless. Correct tail: [return_home] only."
        ),
    ),

    Scenario(
        id="B2",
        group="B",
        signal_type="implicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
                MockCylinder(id=3, world_x=15.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1),
            ],
            remaining_steps=[
                _approach(2), _map_orbit(2),
                _return_home(),
            ],
            failure_context=None,
        ),
        expected_nominal=False,
        expected_tail=[_approach(2), _map_orbit(2), _approach(3), _map_orbit(3), _return_home()],
        note=(
            "Three cylinders in manifest. Remaining plan only covers cylinder 2. "
            "Cylinder 3 is absent from the plan entirely. No failure event."
        ),
        rationale=(
            "Three cylinders in manifest. Remaining plan only covers cylinder 2. "
            "Cylinder 3 is absent entirely. Replanner must add approach(3)+map(3) "
            "before return_home."
        ),
    ),

    Scenario(
        id="B3",
        group="B",
        signal_type="implicit",
        mission="Do a thorough search, then map every cylinder you find twice.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=12.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 6.0}),
                _step("search",  {"pattern": "lawnmower"}),
            ],
            remaining_steps=[
                _approach("all"),
                _map_orbit("all", repeat=2),
                _return_home(),
            ],
            failure_context=None,
        ),
        expected_nominal=False,
        expected_tail=[
            _approach(1), _map_orbit(1, repeat=2),
            _approach(2), _map_orbit(2, repeat=2),
            _return_home(),
        ],
        note=(
            "Remaining plan uses 'all' placeholder. Two specific cylinders now "
            "known. Replanner should expand to explicit cylinder IDs with repeat=2."
        ),
        rationale=(
            "Remaining plan uses 'all' placeholder. Two specific cylinders (1 and 2) "
            "are now known. Replanner should expand 'all' to explicit per-cylinder "
            "approach+map(repeat=2) pairs."
        ),
    ),

    Scenario(
        id="B4",
        group="B",
        signal_type="implicit",
        mission="Find any cylinders and map each one 3 times.",
        world=MockWorldState(
            cylinders=[MockCylinder(id=1, world_x=7.0, world_y=0.0, depth_m=6.0)],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1, repeat=3),
            ],
            remaining_steps=[_return_home()],
            failure_context=None,
        ),
        expected_nominal=True,
        expected_tail=None,
        note=(
            "3 orbits complete on the only cylinder. Only return_home remains. "
            "Correct answer is NOMINAL. Tests false-replan rate."
        ),
        rationale=(
            "3 orbits complete on the only cylinder. Only return_home remains. "
            "Correct answer is NOMINAL. Tests false-replan rate."
        ),
    ),

    # =========================================================================
    # GROUP C – Battery telemetry only
    # failure_context gives the raw reading; replanner decides what to do
    # =========================================================================

    Scenario(
        id="C1",
        group="C",
        signal_type="explicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
                _approach(1), _map_orbit(1),
            ],
            remaining_steps=[_approach(2), _map_orbit(2), _return_home()],
            failure_context="Battery telemetry reports 15% remaining.",
            battery_pct=15.0,
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        note=(
            "Battery at 15%. Replanner receives raw telemetry only. "
            "Must decide independently whether to continue or abort."
        ),
        rationale=(
            "Battery at 15% is critically low. Replanner must abort the remaining "
            "cylinder 2 work and return home immediately."
        ),
    ),

    Scenario(
        id="C2",
        group="C",
        signal_type="explicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[MockCylinder(id=1, world_x=5.0, world_y=0.0, depth_m=5.0)],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "yaw_scan"}),
            ],
            remaining_steps=[_approach(1), _map_orbit(1), _return_home()],
            failure_context="Battery telemetry reports 55% remaining.",
            battery_pct=55.0,
        ),
        expected_nominal=True,
        expected_tail=None,
        note=(
            "Battery at 55%. Remaining plan is reasonable. "
            "Correct answer is NOMINAL. Tests whether replanner overreacts to "
            "any battery mention when level is not actually critical."
        ),
        rationale=(
            "Battery at 55% is sufficient to complete the remaining plan. "
            "Correct answer is NOMINAL. Tests whether replanner overreacts to "
            "any battery mention when level is not actually critical."
        ),
    ),

    Scenario(
        id="C3",
        group="C",
        signal_type="explicit",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[
                MockCylinder(id=1, world_x=5.0,  world_y=0.0, depth_m=5.0),
                MockCylinder(id=2, world_x=10.0, world_y=0.0, depth_m=5.0),
                MockCylinder(id=3, world_x=15.0, world_y=0.0, depth_m=5.0),
            ],
            completed_steps=[
                _step("takeoff", {"altitude": 5.0}),
                _step("search",  {"pattern": "lawnmower"}),
                _approach(1), _map_orbit(1),
            ],
            remaining_steps=[
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
            failure_context="Battery telemetry reports 20% remaining.",
            battery_pct=20.0,
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        note=(
            "Battery at 20% with two full approach+map pairs remaining. "
            "Replanner must judge whether the workload is feasible."
        ),
        rationale=(
            "Battery at 20% with two full approach+map pairs remaining is "
            "insufficient. Replanner must abort and return home immediately."
        ),
    ),
]


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(model_keys: list[str], output_dir: str = "."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list] = {mk: [] for mk in model_keys}

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: REPLANNING UNDER FAILURE")
    print("=" * 70)
    print()
    print("Evaluation framework (mirrors Experiments 3 & 4):")
    print("  behavior_accuracy [PRIMARY] = correct NOMINAL/REPLAN decision AND")
    print("                               (for REPLAN) tail matches reference exactly")
    print("  mission_success   [exp1-parallel] = % of ALL missions fully correct")
    print("  decision_accuracy           = binary NOMINAL/REPLAN accuracy (TP+TN rate)")
    print("  Confusion matrix: REPLAN is the positive class.")
    print()
    print("Signal types:")
    print("  explicit = raw sensor/perception/telemetry event in failure_context")
    print("  implicit = no failure_context; replanner reasons from manifest alone")
    print("  history  = no failure_context; replanner reasons from completion record")
    print()

    for scenario in SCENARIOS:
        print(f"\n--- [{scenario.id}] Group {scenario.group} [{scenario.signal_type}]: {scenario.note[:60]}...")
        exp_str = "NOMINAL" if scenario.expected_nominal else "REPLAN"
        print(f"    Expected: {exp_str}  |  {scenario.rationale[:65]}...")

        for model_key in model_keys:
            print(f"  Model: {model_key}")

            result = run_replan(
                mission_intent=scenario.mission,
                world=scenario.world,
                model_key=model_key,
            )

            # Tail matching
            if not result["nominal"] and scenario.expected_tail is not None:
                tail_match, tail_match_detail = _tails_match(
                    result.get("tail"), scenario.expected_tail
                )
            elif result["nominal"] and scenario.expected_tail is None:
                tail_match        = None
                tail_match_detail = "NOMINAL — no tail to compare."
            elif result["nominal"] and scenario.expected_tail is not None:
                tail_match        = False
                tail_match_detail = "Replanner said NOMINAL; reference tail exists."
            else:
                tail_match, tail_match_detail = None, "No reference tail — content check skipped."

            # Confusion matrix classification
            decision_class = _classify_decision(result["nominal"], scenario.expected_nominal)

            # Composite behavior accuracy
            behavior_correct = _check_behavior_accuracy(
                nominal=result["nominal"],
                expected_nominal=scenario.expected_nominal,
                valid=result["valid"],
                tail_match=tail_match,
            )

            # Mission success (exp1-parallel)
            mission_success = _check_mission_success(
                nominal=result["nominal"],
                expected_nominal=scenario.expected_nominal,
                valid=result["valid"],
                tail_match=tail_match,
            )

            # Steps preserved (secondary diagnostic)
            preserved = steps_preserved_pct(
                scenario.world.completed_steps,
                result.get("tail"),
            )

            result["scenario_id"]       = scenario.id
            result["scenario_group"]    = scenario.group
            result["signal_type"]       = scenario.signal_type
            result["expected_nominal"]  = scenario.expected_nominal
            result["expected_tail"]     = scenario.expected_tail
            result["decision_class"]    = decision_class
            result["behavior_correct"]  = behavior_correct
            result["mission_success"]   = mission_success
            result["tail_match"]        = tail_match
            result["tail_match_detail"] = tail_match_detail
            result["steps_preserved"]   = preserved

            all_results[model_key].append(result)

            status   = "NOMINAL" if result["nominal"] else ("VALID" if result["valid"] else "INVALID")
            beh_mark = ("✓" if behavior_correct else "✗") if behavior_correct is not None else "?"
            ms_mark  = ("✓" if mission_success  else "✗") if mission_success  is not None else "?"
            dc_str   = f"[{decision_class}]" if decision_class else "[?]"
            tm_str   = ""
            if tail_match is not None:
                tm_str = f"  tail:{'✓' if tail_match else '✗'}"

            print(
                f"    {status}  {dc_str}  [behav:{beh_mark}]  [mission:{ms_mark}]{tm_str}  "
                f"attempts={result['attempts']}  "
                f"latency={result['latency_s']:.1f}s"
            )
            if tail_match is False and not result["nominal"]:
                print(f"      tail_mismatch: {tail_match_detail}")
                if scenario.expected_tail:
                    ref_states  = [s.get("state") for s in scenario.expected_tail]
                    prod_states = [s.get("state") for s in (result.get("tail") or [])]
                    print(f"      reference: {ref_states}")
                    print(f"      produced:  {prod_states}")

    # ------------------------------------------------------------------
    # Aggregate summary — overall + broken down by signal type
    # ------------------------------------------------------------------
    summary: dict[str, dict] = {}
    signal_types = ["explicit", "implicit", "history"]

    def _agg(recs: list[dict]) -> dict:
        nr = len(recs)
        if nr == 0:
            return {}

        # Confusion matrix (REPLAN = positive class)
        cm: dict[str, int] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for r in recs:
            dc = r.get("decision_class")
            if dc in cm:
                cm[dc] += 1

        n_labeled_dec    = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        decision_correct = cm["TP"] + cm["TN"]

        n_replan_expected  = cm["TP"] + cm["FN"]
        n_nominal_expected = cm["TN"] + cm["FP"]

        precision = cm["TP"] / max(cm["TP"] + cm["FP"], 1)
        recall    = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr       = cm["FP"] / max(n_nominal_expected, 1)
        fnr       = cm["FN"] / max(n_replan_expected,  1)

        # Composite behavior accuracy (PRIMARY)
        labeled_beh   = [r for r in recs if r.get("behavior_correct") is not None]
        n_labeled_beh = len(labeled_beh)
        behavior_ok   = sum(1 for r in labeled_beh if r["behavior_correct"])

        # Mission success (exp1-parallel) — denominator is ALL missions
        labeled_ms   = [r for r in recs if r.get("mission_success") is not None]
        n_labeled_ms = len(labeled_ms)
        mission_ok   = sum(1 for r in labeled_ms if r["mission_success"])

        # Tail match stats
        tail_check_recs = [r for r in recs if r.get("tail_match") is not None]
        n_tail_checks   = len(tail_check_recs)
        n_tail_match_ok = sum(1 for r in tail_check_recs if r["tail_match"])

        # Steps preserved
        pres_vals = [r["steps_preserved"] for r in recs if r.get("steps_preserved") is not None]

        return {
            "n":                     nr,
            "behavior_accuracy_pct": round(100 * behavior_ok  / max(n_labeled_beh, 1), 1),
            "mission_success_pct":   round(100 * mission_ok   / max(n_labeled_ms,  1), 1),
            "decision_accuracy_pct": round(100 * decision_correct / max(n_labeled_dec, 1), 1),
            "cm_TP": cm["TP"], "cm_FP": cm["FP"],
            "cm_TN": cm["TN"], "cm_FN": cm["FN"],
            "true_positive_rate":    round(100 * recall,    1),
            "true_negative_rate":    round(100 * cm["TN"] / max(n_nominal_expected, 1), 1),
            "false_positive_rate":   round(100 * fpr,       1),
            "false_negative_rate":   round(100 * fnr,       1),
            "precision":             round(100 * precision,  1),
            "f1_score":              round(100 * f1,         1),
            "tail_match_pct":        round(100 * n_tail_match_ok / n_tail_checks, 1) if n_tail_checks else None,
            "n_tail_checks":         n_tail_checks,
            "steps_preserved_pct":   round(sum(pres_vals) / len(pres_vals), 1) if pres_vals else None,
            "avg_attempts":          round(sum(r["attempts"]  for r in recs) / nr, 2),
            "avg_latency_s":         round(sum(r["latency_s"] for r in recs) / nr, 2),
        }

    for model_key in model_keys:
        records   = all_results[model_key]
        by_signal = {
            st: _agg([r for r in records if r["signal_type"] == st])
            for st in signal_types
        }
        summary[model_key] = {"overall": _agg(records), "by_signal_type": by_signal}

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("EXPERIMENT 2 SUMMARY — Table 1: Primary Outcomes (Overall)")
    print()
    print("  Confusion matrix uses REPLAN as the positive class:")
    print("    TP = expected REPLAN, said REPLAN")
    print("    FP = expected NOMINAL, said REPLAN   (spurious change)")
    print("    TN = expected NOMINAL, said NOMINAL  (correct quiescence)")
    print("    FN = expected REPLAN, said NOMINAL   (missed failure — dangerous)")
    print()
    print("  behav%   = behavior_accuracy_pct [PRIMARY] — correct decision AND")
    print("             (for REPLAN) produced tail matched reference tail exactly.")
    print("  mission% = mission_success_pct [exp1-parallel] — % of ALL missions")
    print("             (NOMINAL + REPLAN) where outcome was fully correct.")
    print("             Mirrors perfect_tail_pct from Experiment 1.")
    print("  dec%     = decision_accuracy_pct — binary NOMINAL/REPLAN only (TP+TN rate).")
    print("             Gap between dec% and behav% = tail-content error rate.")
    print("  tail%    = % of REPLAN scenarios where tail matched reference exactly.")
    print("  FPR%     = false_positive_rate  (FP / expected NOMINAL).")
    print("  FNR%     = false_negative_rate  (FN / expected REPLAN).")
    print("=" * 100)
    cols = ["behav%", "mission%", "dec%", "tail%", "FPR%", "FNR%", "F1%", "attempts", "lat(s)"]
    print(f"{'Model':<22} " + "  ".join(f"{c:>9}" for c in cols))
    print("-" * 105)
    for mk, s in summary.items():
        o  = s["overall"]
        tm = o.get("tail_match_pct")
        print(
            f"{mk:<22}  "
            f"{o.get('behavior_accuracy_pct', 0):>9.1f}  "
            f"{o.get('mission_success_pct',   0):>9.1f}  "
            f"{o.get('decision_accuracy_pct', 0):>9.1f}  "
            f"{'N/A' if tm is None else f'{tm:.1f}':>9}  "
            f"{o.get('false_positive_rate',   0):>9.1f}  "
            f"{o.get('false_negative_rate',   0):>9.1f}  "
            f"{o.get('f1_score',              0):>9.1f}  "
            f"{o.get('avg_attempts',          0):>9.2f}  "
            f"{o.get('avg_latency_s',         0):>8.2f}"
        )

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 2 SUMMARY — Table 2: Confusion Matrix (Overall, REPLAN = positive class)")
    print()
    print("  TPR% = recall = TP/(TP+FN)   TNR% = TN/(TN+FP)")
    print("  FPR% = FP/(TN+FP)            FNR% = FN/(TP+FN)")
    print("  Prec = TP/(TP+FP)            F1   = 2·Prec·Recall/(Prec+Recall)")
    print("=" * 100)
    cols_a = ["TP", "FP", "TN", "FN", "TPR%", "TNR%", "FPR%", "FNR%", "Prec%", "F1%"]
    print(f"{'Model':<22} " + "  ".join(f"{c:>6}" for c in cols_a))
    print("-" * 95)
    for mk, s in summary.items():
        o = s["overall"]
        print(
            f"{mk:<22}  "
            f"{o.get('cm_TP',              0):>6}  "
            f"{o.get('cm_FP',              0):>6}  "
            f"{o.get('cm_TN',              0):>6}  "
            f"{o.get('cm_FN',              0):>6}  "
            f"{o.get('true_positive_rate', 0):>6.1f}  "
            f"{o.get('true_negative_rate', 0):>6.1f}  "
            f"{o.get('false_positive_rate',0):>6.1f}  "
            f"{o.get('false_negative_rate',0):>6.1f}  "
            f"{o.get('precision',          0):>6.1f}  "
            f"{o.get('f1_score',           0):>6.1f}"
        )

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 2 SUMMARY — Table 3: Breakdown by Signal Type")
    print("  explicit = raw sensor/perception/telemetry event")
    print("  implicit = replanner reasons from manifest alone")
    print("  history  = replanner reasons from completion record alone")
    print("=" * 100)
    for st in signal_types:
        print(f"\n  Signal type: {st}")
        print(f"  {'Model':<22} " + "  ".join(f"{c:>9}" for c in cols))
        print("  " + "-" * 105)
        for mk, s in summary.items():
            b = s["by_signal_type"].get(st, {})
            if not b:
                print(f"  {mk:<22}  (no scenarios)")
                continue
            tm = b.get("tail_match_pct")
            print(
                f"  {mk:<22}  "
                f"{b.get('behavior_accuracy_pct', 0):>9.1f}  "
                f"{b.get('mission_success_pct',   0):>9.1f}  "
                f"{b.get('decision_accuracy_pct', 0):>9.1f}  "
                f"{'N/A' if tm is None else f'{tm:.1f}':>9}  "
                f"{b.get('false_positive_rate',   0):>9.1f}  "
                f"{b.get('false_negative_rate',   0):>9.1f}  "
                f"{b.get('f1_score',              0):>9.1f}  "
                f"{b.get('avg_attempts',          0):>9.2f}  "
                f"{b.get('avg_latency_s',         0):>8.2f}"
            )
            cm = {k: b.get(f"cm_{k}", 0) for k in ("TP", "FP", "TN", "FN")}
            print(f"    confusion: TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}")

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 2 SUMMARY — Table 4: Per-Scenario Decision Matrix")
    print("  behav = ✓ (correct decision + tail match) or ✗.")
    print("  mission = ✓ (fully correct outcome, exp1-parallel) or ✗.")
    print("  DC = confusion matrix cell (TP/FP/TN/FN).")
    print("=" * 100)
    hdr = ["Scenario", "Group", "Signal", "Expected"] + model_keys
    print("  " + "  ".join(f"{h:<22}" for h in hdr))
    print("  " + "-" * (24 * len(hdr)))
    for scenario in SCENARIOS:
        exp_str = "NOMINAL" if scenario.expected_nominal else "REPLAN"
        row = [scenario.id, scenario.group, scenario.signal_type[:8], exp_str]
        for mk in model_keys:
            recs = [r for r in all_results[mk] if r["scenario_id"] == scenario.id]
            if recs:
                r   = recs[0]
                beh = ("✓" if r["behavior_correct"] else "✗") if r["behavior_correct"] is not None else "?"
                ms  = ("✓" if r["mission_success"]  else "✗") if r["mission_success"]  is not None else "?"
                dc  = r.get("decision_class", "?")
                tm  = ""
                if r.get("tail_match") is not None:
                    tm = " t:" + ("✓" if r["tail_match"] else "✗")
                row.append(f"b{beh} m{ms} [{dc}]{tm}")
            else:
                row.append("—")
        print("  " + "  ".join(f"{v:<22}" for v in row))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_json(f"{output_dir}/exp2_results_raw.json", all_results)
    save_json(f"{output_dir}/exp2_summary.json", summary)
    print(f"\n  Saved → {output_dir}/")
    print("\nDone.\n")
    return all_results, summary


if __name__ == "__main__":
    MODELS_TO_TEST = [
        "gemini-flash-2.5",
        "qwen-235b",
        "llama-3.1-8b",
    ]
    run_experiment(MODELS_TO_TEST, output_dir="exp2_output")