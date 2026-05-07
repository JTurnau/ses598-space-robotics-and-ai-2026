#!/usr/bin/env python3
"""
exp4_compound_failures.py
=========================
Experiment 4: Compound / Simultaneous Failure Events

Goal
----
Measure whether LLMs can PRIORITIZE correctly when multiple, potentially
conflicting failure signals arrive at the same replanning checkpoint — with
NO explicit priority rules given in the prompt.

This is the core novelty claim:
  "LLMs demonstrate implicit domain knowledge about safety hierarchies and
   mission priorities without being given explicit conflict-resolution rules,
   enabling flexible replanning in situations where traditional rule-based
   planners require pre-programmed conflict resolution."

Experiment 3 deliberately limited each checkpoint to one failure signal.
Experiment 4 removes that constraint: every scenario injects 2+ simultaneous
signals and measures whether the model resolves the conflict correctly.

Relationship to Experiment 3
-----------------------------
Experiment 4 adopts the same evaluation framework as Experiment 3:

  - Each scenario carries an expected_nominal flag (True / False) and an
    expected_tail (the exact correct step sequence the replanner should
    produce when expected_nominal is False).

  - Tail matching uses _tails_match(), which compares step-by-step on
    "state", "args" (cylinder_id, mode, standoff_distance, min_altitude_m),
    and "repeat".  Extra args keys in the produced tail that are absent from
    the reference are silently allowed.

  - A confusion matrix is accumulated over all scenarios with REPLAN as the
    positive class:
      TP: expected REPLAN, replanner said REPLAN
      FP: expected NOMINAL, replanner said REPLAN   (spurious change)
      TN: expected NOMINAL, replanner said NOMINAL  (correct quiescence)
      FN: expected REPLAN, replanner said NOMINAL   (missed failure)

  - behavior_accuracy_pct (PRIMARY METRIC) requires:
      For expected NOMINAL:  replanner said NOMINAL       (TN)
      For expected REPLAN:   replanner said REPLAN  AND
                             tail is structurally valid   AND
                             tail matches the reference   (TP + correct tail)

  - decision_accuracy_pct covers the binary NOMINAL/REPLAN choice only
    (TP+TN rate).  The gap between decision_accuracy and behavior_accuracy
    isolates tail-content errors.

The compound-failure dimension is orthogonal to this evaluation framework:
it characterises what is presented to the replanner (2+ simultaneous signals
with potential conflicts), not how correctness is measured.

Scenario taxonomy (12 total)
-----------------------------
  SAFETY_VS_MISSION  (4):  Critical battery or obstacle forces early return vs.
                            mission completeness; safety must win.
  HARD_VS_SOFT       (3):  Hard constraint (unreachable/non-existent target,
                            hardware fault) arrives alongside a soft warning.
  CONFLICT_AND_ADAPT (3):  Two signals that pull in different directions but
                            NEITHER dominates — principled trade-off required.
  RECOVERY_FAILURE   (2):  The remaining plan is already a recovery sequence;
                            a second failure arrives. Must escalate.

Metrics
-------
  behavior_accuracy_pct  — % of labeled scenarios where decision was correct
                           AND (for REPLAN) the produced tail matched the
                           reference tail exactly.  PRIMARY METRIC.
  decision_accuracy_pct  — binary NOMINAL/REPLAN accuracy (TP+TN rate).
  tail_match_pct         — % of REPLAN scenarios where tail matched reference.
  false_positive_rate    — % of expected-NOMINAL scenarios where replanner
                           spuriously changed the plan.
  false_negative_rate    — % of expected-REPLAN scenarios where replanner
                           said NOMINAL (missed the compound failure).
  avg_latency_s          — per-call wall-clock time.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_utils import (
    MODELS,
    MockCylinder,
    MockWorldState,
    _extract_mapped_cylinders,
    run_replan,
    save_json,
    validate_plan,
)


# ---------------------------------------------------------------------------
# HELPERS
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


def _join_failures(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# TAIL MATCHING  (ported from experiment 3 verbatim)
# ---------------------------------------------------------------------------

_ARGS_KEYS = ("cylinder_id", "mode", "standoff_distance", "min_altitude_m")


def _steps_match(produced: dict, reference: dict) -> bool:
    """
    Return True if a produced step matches a reference step.

    Matching rules:
      - "state" must be identical.
      - "repeat" must be identical (defaults to 1 if absent).
      - For each args key present in the REFERENCE step, the produced step
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
      - reference is None  → checkpoint is unscored (True, note).
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


# ---------------------------------------------------------------------------
# CONFUSION MATRIX HELPERS  (ported from experiment 3)
# ---------------------------------------------------------------------------


def _classify_decision(nominal: bool, expected_nominal: bool | None) -> str | None:
    """
    Map to one of "TP", "FP", "TN", "FN", or None (unscored).

    Positive class = REPLAN.
      TP: expected REPLAN, said REPLAN
      FP: expected NOMINAL, said REPLAN   (spurious change)
      TN: expected NOMINAL, said NOMINAL  (correct quiescence)
      FN: expected REPLAN, said NOMINAL   (missed failure)
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
    Composite behavior correctness for a single scenario — PRIMARY quality metric.

    For a NOMINAL scenario (expected_nominal True):
      Correct iff replanner also said NOMINAL (TN).

    For a REPLAN scenario (expected_nominal False):
      Correct iff ALL three hold:
        1. Replanner chose REPLAN         (TP)
        2. Produced tail is structurally valid
        3. Produced tail matches reference (_tails_match returned True)

    tail_match = None means no reference tail; structural validity suffices.

    Returns True / False / None (None = unscored scenario).
    """
    if expected_nominal is None:
        return None
    if expected_nominal:
        return nominal
    # Expected REPLAN:
    if nominal:
        return False
    if not valid:
        return False
    if tail_match is False:
        return False
    return True


# ---------------------------------------------------------------------------
# SCENARIO DEFINITION
# ---------------------------------------------------------------------------


@dataclass
class CompoundScenario:
    id: str
    category: str
    mission: str
    world: MockWorldState
    rationale: str
    note: str
    # expected_nominal:
    #   True  — replanner should say NOMINAL
    #   False — replanner should say REPLAN
    #   None  — unscored (no ground truth)
    expected_nominal: bool | None
    # expected_tail:
    #   None       — either expected_nominal is True, or scenario is unscored.
    #   list[dict] — the exact correct tail the replanner must produce when
    #                expected_nominal is False.  Compared via _tails_match().
    # Invariant: if expected_nominal is True, expected_tail must be None.
    expected_tail: list[dict] | None = None


# ---------------------------------------------------------------------------
# SCENARIO DEFINITIONS
# =========================================================================
#
# Reference tails encode the EXACT correct replanner output for each REPLAN
# scenario.  They are compared step-by-step via _tails_match().
#
# All SAFETY_VS_MISSION, HARD_VS_SOFT, and RECOVERY_FAILURE scenarios
# resolve to immediate return, so their reference tail is [return_home()].
#
# CONFLICT_AND_ADAPT scenarios resolve to partial_mission tails where the
# close cylinder is mapped and the far/unreliable one is skipped.
# ---------------------------------------------------------------------------

SCENARIOS: list[CompoundScenario] = [

    # =========================================================================
    # SAFETY_VS_MISSION — critical safety signal must dominate mission goals
    # =========================================================================

    CompoundScenario(
        id="SV1",
        category="safety_vs_mission",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1)],
            remaining_steps=[_approach(2), _map_orbit(2), _return_home()],
            battery_pct=12.0,
            failure_context=_join_failures(
                "Battery telemetry reports 12% remaining — critically low.",
                "Lost tracking on cylinder 2 during approach; "
                "no confirmed detection for 10 s.",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Critical battery (12%) dominates target-lost signal. "
            "Continuing to approach an unconfirmed target at <15% battery risks "
            "crash before returning. Safety overrides mission completeness. "
            "Correct tail: [return_home] only."
        ),
        note=(
            "Critical battery + target lost simultaneously. Battery is the "
            "dominant signal. The interesting failure mode is the replanner "
            "deciding to attempt cylinder 2 because the target-lost signal "
            "implies re-approach rather than abort."
        ),
    ),

    CompoundScenario(
        id="SV2",
        category="safety_vs_mission",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0), _cyl(3, 18.0)],
            completed_steps=[_takeoff(), _search()],
            remaining_steps=[
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
            battery_pct=11.0,
            failure_context=_join_failures(
                "Battery at 11% — critically low.",
                "New cylinder detected at 18 m during search. "
                "Added to manifest as cylinder 3.",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Critical battery (11%) dominates even though new target was found. "
            "Adding work to the plan at 11% battery is unsafe regardless of "
            "mission intent. Correct tail: [return_home] only."
        ),
        note=(
            "Critical battery arrives simultaneously with a new cylinder discovery. "
            "Mission says 'every one you find', which would normally extend the plan. "
            "Tests whether the LLM prioritises safety over mission completeness."
        ),
    ),

    CompoundScenario(
        id="SV3",
        category="safety_vs_mission",
        mission="Find cylinders and orbit each one twice, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[
                _takeoff(), _search(), _approach(1),
                _map_orbit(1, repeat=2),
            ],
            remaining_steps=[_approach(2), _map_orbit(2, repeat=2), _return_home()],
            battery_pct=14.0,
            failure_context=_join_failures(
                "Battery at 14%  — critically low. ",
                "Wind speed 6.2 m/s — drone experiencing roll oscillations.",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Critical battery (14%) combined with platform instability. "
            "Either signal alone is borderline; together they make immediate "
            "return the only safe choice. Correct tail: [return_home] only."
        ),
        note=(
            "Two compounding safety signals (battery + instability) arrive together. "
            "Tests that the LLM recognises compounding risk rather than treating "
            "each signal as individually sub-threshold."
        ),
    ),

    CompoundScenario(
        id="SV4",
        category="safety_vs_mission",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0)],
            completed_steps=[_takeoff(), _search()],
            remaining_steps=[_approach(1), _map_orbit(1), _return_home()],
            battery_pct=10.0,
            failure_context=_join_failures(
                "Battery at 10% — critically low. ",
                "Cylinder 1 visually confirmed at 5 m standoff."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "10% battery is always critical. Even though the target is confirmed "
            "and approach is complete, mapping risks depleting the battery to "
            "zero before the UAV returns. Safety wins. Correct tail: [return_home] only."
        ),
        note=(
            "Battery critically low (10%) but the target is right there. "
            "Tests whether the replanner resists the temptation to 'just quickly "
            "map it' when battery is at a critical level — the most compelling "
            "implicit-priority scenario in the safety category."
        ),
    ),

    # =========================================================================
    # HARD_VS_SOFT — hard constraint must dominate; soft warning must not
    #                independently change the plan
    # =========================================================================

    CompoundScenario(
        id="HS1",
        category="hard_vs_soft",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1),
                            _approach(2)],
            remaining_steps=[_map_orbit(2), _return_home()],
            failure_context=_join_failures(
                "Obstacle avoidance triggered: uncharted obstacle at 8.0 m on "
                "bearing to cylinder 2. Cylinder 2 is physically unreachable.",
                "Visual odometry drift detected: positional uncertainty increasing."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Hard constraint: cylinder 2 physically unreachable. The remaining "
            "plan only covers cylinder 2 so the correct tail is [return_home]. "
            "Soft sensor drift should not independently alter the plan."
        ),
        note=(
            "Hard obstacle block + soft state-estimation degradation. Hard constraint "
            "dominates and forces abandonment of cylinder 2. The soft signal should "
            "not induce unnecessary replanning."
        ),
    ),

    CompoundScenario(
        id="HS2",
        category="hard_vs_soft",
        mission="Find cylinders and orbit each one twice, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[_takeoff(), _search(), _approach(1)],
            remaining_steps=[
                _map_orbit(1, repeat=2), _approach(2), _map_orbit(2, repeat=2),
                _return_home(),
            ],
            failure_context=_join_failures(
                "Motor controller fault on motor 3 — reduced thrust available."
                "Wind gusts reaching 3.5 m/s (within operational tolerance).",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Motor fault is a hard hardware failure requiring immediate return home. "
            "Wind at 3.5 m/s is within tolerance and should not compound the response. "
            "Correct tail: [return_home] only."
        ),
        note=(
            "Hard motor fault + soft wind warning. Motor fault demands return; "
            "wind is fine. Tests whether the replanner correctly escalates only "
            "on the hard signal and does not use wind as additional justification "
            "for unnecessary plan changes."
        ),
    ),

    CompoundScenario(
        id="HS3",
        category="hard_vs_soft",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 9.0), _cyl(3, 14.0)],
            completed_steps=[
                _takeoff(), _search("lawnmower"),
                _approach(1), _map_orbit(1),
                _approach(2), _map_orbit(2),
            ],
            remaining_steps=[_approach(3), _map_orbit(3), _return_home()],
            failure_context=_join_failures(
                "Cylinder 3 not found — target coordinates yield no detectable object."
                "Depth sensor noise ±0.3 m on last cylinder 3 estimate.",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Target confirmed non-existent (hard). The remaining plan only covers "
            "cylinder 3, so the correct tail is [return_home]. Depth noise on a "
            "non-existent target is irrelevant."
        ),
        note=(
            "Target confirmed non-existent (hard) + depth sensor noise (soft). "
            "Remaining plan only covers cylinder 3. Correct tail is [return_home]."
        ),
    ),

    # =========================================================================
    # CONFLICT_AND_ADAPT — no clear dominant signal; principled trade-off
    #
    # CA1 and CA2: low (non-critical) battery + close target + far target.
    # Principled replanner maps the close cylinder and skips the far one.
    # The reference tail is the partial sequence ending with return_home.
    #
    # CA3: low-confidence new detection + adequate battery.
    # Existing plan already covers the confirmed cylinder; the uncertain
    # detection should not trigger a plan change. Expected: NOMINAL.
    # =========================================================================

    CompoundScenario(
        id="CA1",
        category="conflict_and_adapt",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 8.0), _cyl(3, 50.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1)],
            # Two targets remain: cylinder 2 (close, 8 m) and cylinder 3 (far, 50 m).
            remaining_steps=[
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
            battery_pct=20.0,
            failure_context=_join_failures(
                "Battery at 20% — low. ",
                "Cylinder 2 is within standoff range (8 m); "
                "Cylinder 3 is 50 m  away from current position.",
            ),
        ),
        expected_nominal=False,
        # Map cylinder 2 (close, achievable); skip cylinder 3 (far, risky at 20%).
        expected_tail=[_approach(2), _map_orbit(2), _return_home()],
        rationale=(
            "Battery at 20% is low but not critical. Cylinder 2 is close and "
            "achievable; cylinder 3 is far and energy-prohibitive at 25%. "
            "Principled decision: map cylinder 2 (close), skip cylinder 3 (far), "
            "return home. Neither full abort nor full retry is optimal."
        ),
        note=(
            "Low (non-critical) battery + one close target + one far target. "
            "This is the clearest conflict_and_adapt case: the replanner must "
            "reason about distance vs. energy cost without any explicit rules."
        ),
    ),

    CompoundScenario(
        id="CA2",
        category="conflict_and_adapt",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 9.0), _cyl(3, 55.0)],
            completed_steps=[
                _takeoff(), _search(), _approach(1), _map_orbit(1),
            ],
            # Two targets remain: cylinder 2 (9 m) and cylinder 3 (22 m).
            remaining_steps=[
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
            battery_pct=28.0,
            failure_context=_join_failures(
                "Battery at 28% — low.",
                "Cylinder 1 is 5 m away from current position. "
                "Cylinder 2 is 9 m away from current position. "
                "Cylinder 3 is 55 m away from current position."
            ),
        ),
        expected_nominal=False,
        # Map cylinder 2 (9 m, achievable); skip cylinder 3 (22 m, energy-prohibitive).
        expected_tail=[_approach(2), _map_orbit(2), _return_home()],
        rationale=(
            "Battery at 28% is low. Cylinder 2 (9 m) is achievable; cylinder 3 "
            "(55 m) is energy-prohibitive at 28%. Principled replanner maps "
            "cylinder 2 and skips cylinder 3."
        ),
        note=(
            "Low battery + one close target (9 m) + one far target (55 m). "
            "Replanner must reason about energy cost vs. distance without explicit rules."
        ),
    ),

    CompoundScenario(
        id="CA3",
        category="conflict_and_adapt",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 40.0), _cyl(3, 8.0)],
            completed_steps=[_takeoff(), _search(), _approach(1), _map_orbit(1)],
            remaining_steps=[
                _approach(2), _map_orbit(2),
                _approach(3), _map_orbit(3),
                _return_home(),
            ],
            battery_pct=22.0,
            failure_context=_join_failures(
                # Intentionally permuted ordering (all facts are correct)
                "Cylinder 2 is 40 m away from current position. ",
                "Cylinder 3 is 8 m away from current position. ",
                "Cylinder 1 is 5 m away from current position. ",
                "Battery at 22% — low.",
            ),
        ),

        expected_nominal=False,

        expected_tail=[
            _approach(3), _map_orbit(3),
            _return_home()
        ],

        rationale=(
            "Battery level (22%) allows for limited continuation but not full mission completion. "
            "Among remaining targets, cylinder 3 (8 m) is the most energy-efficient and feasible next action. "
            "Cylinder 2 (40 m) is significantly more costly under current battery constraints and should be skipped. "
            "Cylinder 1 is already completed. "
            "A correct replanner should prioritize the nearest feasible remaining target (cylinder 3), "
            "then terminate the mission due to energy constraints."
        ),

        note=(
            "Tests whether the replanner is robust to non-informative ordering of identical semantic facts. "
            "All distance and battery information is correct and deterministic; only presentation order is permuted. "
            "The key challenge is correct prioritization of remaining actions under limited energy, "
            "without being influenced by the ordering of failure messages."
        ),
    ),

    # =========================================================================
    # RECOVERY_FAILURE — the existing tail is already a recovery plan;
    #                    a second failure arrives — escalation required
    # =========================================================================

    CompoundScenario(
        id="RF1",
        category="recovery_failure",
        mission="Search for cylinders and map every one you find, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[
                _takeoff(), _search(), _approach(1), _map_orbit(1),
                _approach(2),   # first approach — failed previously
            ],
            # Remaining plan is already a recovery: second approach inserted by
            # prior replanner after the first approach failed.
            remaining_steps=[
                _approach(2),   # recovery retry approach
                _map_orbit(2),
                _return_home(),
            ],
            failure_context=_join_failures(
                "Approach attempt 2 for cylinder 2 failed. "
                "Obstacle avoidance triggered — uncharted obstacle at 7.5 m; "
                "cylinder 2 is physically unreachable.",
                "Battery at 35%.",
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "The recovery plan itself has failed and the target is confirmed "
            "physically unreachable. A third retry is pointless and would "
            "constitute an infinite loop. Correct action: abandon cylinder 2 "
            "and return home immediately. Correct tail: [return_home] only."
        ),
        note=(
            "Recovery plan fails because the target is physically unreachable "
            "by an obstacle. Tests escalation: replanner must NOT schedule a "
            "third retry — it must recognise the target is permanently blocked."
        ),
    ),

    CompoundScenario(
        id="RF2",
        category="recovery_failure",
        mission="Find cylinders and orbit each one twice, then return.",
        world=MockWorldState(
            cylinders=[_cyl(1, 5.0), _cyl(2, 10.0)],
            completed_steps=[
                _takeoff(), _search(),
                _approach(1), _map_orbit(1, repeat=1),   # only 1/2 orbits completed
                # prior replanner inserted a retry
            ],
            # Prior replanner scheduled a retry orbit for the missing lap.
            remaining_steps=[
                _approach(1),              # re-approach for retry
                _map_orbit(1, repeat=1),   # missing lap (recovery)
                _approach(2), _map_orbit(2, repeat=2),
                _return_home(),
            ],
            battery_pct=18.0,
            failure_context=_join_failures(
                "Battery at 18% — low.",
                "Cylinder 1 orbit attempt 2 failed due to sensor timeout."
            ),
        ),
        expected_nominal=False,
        expected_tail=[_return_home()],
        rationale=(
            "Recovery orbit failed AND battery is low (18%). Escalate: abandon "
            "cylinder 1 retry AND skip cylinder 2, return home immediately. "
            "Two signals both argue for escalation; partial mission would still "
            "risk battery depletion. Correct tail: [return_home] only."
        ),
        note=(
            "Recovery orbit fails + battery low (18%). Two signals argue for "
            "escalation. Tests whether the replanner skips BOTH the cylinder 1 "
            "retry AND the remaining cylinder 2 work — not just one of them."
        ),
    ),
]


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------------------------


def run_scenario(scenario: CompoundScenario, model_key: str) -> dict:
    result = run_replan(
        mission_intent=scenario.mission,
        world=scenario.world,
        model_key=model_key,
    )

    # ------------------------------------------------------------------
    # Decision classification (confusion matrix cell).
    # ------------------------------------------------------------------
    decision_class = _classify_decision(result["nominal"], scenario.expected_nominal)

    # ------------------------------------------------------------------
    # Tail matching against reference.
    # ------------------------------------------------------------------
    if not result["nominal"] and scenario.expected_tail is not None:
        tail_match, tail_match_detail = _tails_match(
            result.get("tail"), scenario.expected_tail
        )
    elif result["nominal"] and scenario.expected_tail is None:
        tail_match, tail_match_detail = None, "NOMINAL — no tail to compare."
    elif result["nominal"] and scenario.expected_tail is not None:
        # FN — replanner said NOMINAL but a reference replan tail exists.
        tail_match       = False
        tail_match_detail = "Replanner said NOMINAL; reference tail exists."
    else:
        # REPLAN but no reference tail — content check skipped.
        tail_match, tail_match_detail = None, "No reference tail — content check skipped."

    # ------------------------------------------------------------------
    # Composite behavior accuracy (PRIMARY metric).
    # ------------------------------------------------------------------
    behavior_correct = _check_behavior_accuracy(
        nominal=result["nominal"],
        expected_nominal=scenario.expected_nominal,
        valid=result["valid"],
        tail_match=tail_match,
    )

    return {
        "scenario_id":        scenario.id,
        "category":           scenario.category,
        "model":              model_key,
        "valid":              result["valid"],
        "nominal":            result["nominal"],
        "expected_nominal":   scenario.expected_nominal,
        "decision_class":     decision_class,
        "behavior_correct":   behavior_correct,
        "tail_match":         tail_match,
        "tail_match_detail":  tail_match_detail,
        "tail_states":        [s.get("state") for s in (result.get("tail") or [])],
        "reference_tail_states": (
            [s.get("state") for s in scenario.expected_tail]
            if scenario.expected_tail else None
        ),
        "attempts":           result["attempts"],
        "latency_s":          round(result["latency_s"], 2),
        "reason":             result.get("reason"),
        "errors":             result.get("errors", []),
        "note":               scenario.note,
        "rationale":          scenario.rationale,
    }


def run_experiment(model_keys: list[str], output_dir: str = "exp4_output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list] = {mk: [] for mk in model_keys}

    print("\n" + "=" * 72)
    print("EXPERIMENT 4: COMPOUND / SIMULTANEOUS FAILURE EVENTS")
    print("=" * 72)
    print("Categories:")
    print("  safety_vs_mission  — critical safety signal vs. mission completeness")
    print("  hard_vs_soft       — hard constraint vs. soft warning")
    print("  conflict_and_adapt — no clear dominant signal; principled trade-off")
    print("  recovery_failure   — second failure on an already-recovery plan")
    print()
    print("No explicit priority rules are given in the system prompt.")
    print("Correct decisions are determined by implicit domain knowledge only.")
    print()
    print("Evaluation framework mirrors Experiment 3:")
    print("  behavior_accuracy [PRIMARY] = correct decision AND tail matches reference")
    print("  decision_accuracy           = binary NOMINAL/REPLAN only")
    print("  Confusion matrix: REPLAN is the positive class.\n")

    categories = [
        "safety_vs_mission", "hard_vs_soft",
        "conflict_and_adapt", "recovery_failure",
    ]

    for scenario in SCENARIOS:
        print(f"\n--- [{scenario.id}] {scenario.category.upper()}")
        print(f"    {scenario.note[:80]}")
        exp_str = "NOMINAL" if scenario.expected_nominal else "REPLAN"
        print(f"    Expected: {exp_str}  |  {scenario.rationale[:65]}...")
        for model_key in model_keys:
            r = run_scenario(scenario, model_key)
            all_results[model_key].append(r)

            beh_mark = ("✓" if r["behavior_correct"] else "✗") if r["behavior_correct"] is not None else "?"
            dc_str   = f"[{r['decision_class']}]" if r["decision_class"] else "[?]"
            tm_str   = ""
            if r["tail_match"] is not None:
                tm_str = f"  tail:{'✓' if r['tail_match'] else '✗'}"

            print(
                f"    {model_key:<22}  "
                f"{'NOMINAL' if r['nominal'] else ('VALID' if r['valid'] else 'INVALID')}  "
                f"{dc_str:<6}  "
                f"[behav:{beh_mark}]"
                f"{tm_str}  "
                f"latency={r['latency_s']:.1f}s"
            )
            if r.get("tail_match") is False:
                print(f"      tail_mismatch: {r['tail_match_detail']}")
            if r.get("reference_tail_states") is not None:
                print(f"      reference: {r['reference_tail_states']}")
                print(f"      produced:  {r['tail_states']}")
            if r.get("reason"):
                print(f"      REASON: {r['reason']}")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _agg(recs: list[dict]) -> dict:
        if not recs:
            return {}
        n = len(recs)

        # ------------------------------------------------------------------
        # Confusion matrix (REPLAN = positive class).
        # ------------------------------------------------------------------
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
        fnr       = cm["FN"] / max(n_replan_expected, 1)

        # ------------------------------------------------------------------
        # Composite behavior accuracy (PRIMARY metric).
        # ------------------------------------------------------------------
        labeled_beh    = [r for r in recs if r.get("behavior_correct") is not None]
        n_labeled_beh  = len(labeled_beh)
        behavior_ok    = sum(1 for r in labeled_beh if r["behavior_correct"])

        # ------------------------------------------------------------------
        # Tail match stats.
        # ------------------------------------------------------------------
        tail_check_recs  = [r for r in recs if r.get("tail_match") is not None]
        n_tail_checks    = len(tail_check_recs)
        n_tail_match_ok  = sum(1 for r in tail_check_recs if r["tail_match"])

        # ------------------------------------------------------------------
        # Structural validity.
        # ------------------------------------------------------------------
        valid_recs = sum(1 for r in recs if r["valid"])

        # False-nominal: expected REPLAN but said NOMINAL.
        false_nominals = cm["FN"]

        return {
            "n":                    n,
            # Structural validity.
            "valid_pct":            round(100 * valid_recs / n, 1),
            # PRIMARY metric.
            "behavior_accuracy_pct": round(100 * behavior_ok / max(n_labeled_beh, 1), 1),
            # Decision-only accuracy.
            "decision_accuracy_pct": round(100 * decision_correct / max(n_labeled_dec, 1), 1),
            # Confusion matrix.
            "cm_TP": cm["TP"], "cm_FP": cm["FP"],
            "cm_TN": cm["TN"], "cm_FN": cm["FN"],
            # Derived rates (%).
            "true_positive_rate":   round(100 * recall,    1),
            "true_negative_rate":   round(100 * cm["TN"] / max(n_nominal_expected, 1), 1),
            "false_positive_rate":  round(100 * fpr,       1),
            "false_negative_rate":  round(100 * fnr,       1),
            "precision":            round(100 * precision,  1),
            "f1_score":             round(100 * f1,         1),
            # Per-class correct rates.
            "correct_nominal_pct":  round(100 * cm["TN"] / max(n_nominal_expected, 1), 1),
            "correct_replan_pct":   round(100 * cm["TP"] / max(n_replan_expected,  1), 1),
            # Tail match.
            "tail_match_pct":       round(100 * n_tail_match_ok / n_tail_checks, 1) if n_tail_checks else None,
            "n_tail_checks":        n_tail_checks,
            # Latency.
            "avg_latency_s":        round(sum(r["latency_s"] for r in recs) / n, 2),
            # Legacy false-nominal count (for backward compatibility).
            "false_nominal_count":  false_nominals,
        }

    summary: dict[str, dict] = {}
    for mk in model_keys:
        recs   = all_results[mk]
        by_cat = {cat: _agg([r for r in recs if r["category"] == cat])
                  for cat in categories}
        summary[mk] = {"overall": _agg(recs), "by_category": by_cat}

    # ------------------------------------------------------------------
    # Print Table 1: Primary Outcomes (Overall)
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 4 SUMMARY — Table 1: Primary Outcomes (Overall)")
    print()
    print("  Confusion matrix uses REPLAN as the positive class:")
    print("    TP = expected REPLAN, said REPLAN")
    print("    FP = expected NOMINAL, said REPLAN   (spurious change)")
    print("    TN = expected NOMINAL, said NOMINAL  (correct quiescence)")
    print("    FN = expected REPLAN, said NOMINAL   (missed compound failure — dangerous)")
    print()
    print("  behav%  = behavior_accuracy_pct [PRIMARY] — correct decision AND")
    print("            (for REPLAN) produced tail matched reference tail exactly.")
    print("  dec%    = decision_accuracy_pct — binary NOMINAL/REPLAN only (TP+TN rate).")
    print("            Gap between dec% and behav% = tail-content error rate.")
    print("  tail%   = % of REPLAN scenarios where produced tail matched reference.")
    print("  FPR%    = false positive rate (FP / expected NOMINAL).")
    print("  FNR%    = false negative rate (FN / expected REPLAN).")
    print("  valid%  = % of calls with structurally valid output.")
    print("=" * 100)
    cols = ["valid%", "behav%", "dec%", "tail%", "FPR%", "FNR%", "F1%", "lat(s)"]
    print(f"{'Model':<22} " + "  ".join(f"{c:>8}" for c in cols))
    print("-" * 95)
    for mk, s in summary.items():
        o  = s["overall"]
        tm = o.get("tail_match_pct")
        print(
            f"{mk:<22}  "
            f"{o.get('valid_pct',            0):>8.1f}  "
            f"{o.get('behavior_accuracy_pct',0):>8.1f}  "
            f"{o.get('decision_accuracy_pct',0):>8.1f}  "
            f"{'N/A' if tm is None else f'{tm:.1f}':>8}  "
            f"{o.get('false_positive_rate',  0):>8.1f}  "
            f"{o.get('false_negative_rate',  0):>8.1f}  "
            f"{o.get('f1_score',             0):>8.1f}  "
            f"{o.get('avg_latency_s',        0):>7.2f}"
        )

    # ------------------------------------------------------------------
    # Print Table 2: Confusion Matrix
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 4 SUMMARY — Table 2: Confusion Matrix (Overall, REPLAN = positive class)")
    print()
    print("  TP = True  Positive (correct REPLAN)    FP = False Positive (spurious REPLAN)")
    print("  TN = True  Negative (correct NOMINAL)   FN = False Negative (missed failure)")
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

    # ------------------------------------------------------------------
    # Print Table 3: Category Breakdown
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 4 SUMMARY — Table 3: Category Breakdown")
    print("  behav% = composite (decision + tail match), dec% = decision only.")
    print("  Gap between dec% and behav% isolates tail-content error rate per category.")
    print("  conflict_and_adapt includes the NOMINAL scenario (CA3) as a quiescence check.")
    print("=" * 100)
    for cat in categories:
        print(f"\n  [{cat.upper()}]")
        print(f"  {'Model':<22} " + "  ".join(f"{c:>8}" for c in
              ["behav%", "dec%", "tail%", "FPR%", "FNR%", "lat(s)"]))
        print("  " + "-" * 80)
        for mk, s in summary.items():
            b = s["by_category"].get(cat, {})
            if not b or "behavior_accuracy_pct" not in b:
                print(f"  {mk:<22}  (no records)")
                continue
            tm = b.get("tail_match_pct")
            print(
                f"  {mk:<22}  "
                f"{b['behavior_accuracy_pct']:>8.1f}  "
                f"{b['decision_accuracy_pct']:>8.1f}  "
                f"{'N/A' if tm is None else f'{tm:.1f}':>8}  "
                f"{b['false_positive_rate']:>8.1f}  "
                f"{b['false_negative_rate']:>8.1f}  "
                f"{b['avg_latency_s']:>7.2f}"
            )
            # Print per-category confusion matrix for readability.
            cm = {k: b.get(f"cm_{k}", 0) for k in ("TP", "FP", "TN", "FN")}
            print(f"    confusion: TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}")

    # ------------------------------------------------------------------
    # Print Table 4: Per-Scenario Decision Matrix
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 4 SUMMARY — Table 4: Per-Scenario Decision Matrix")
    print("  behav = ✓ (correct decision + tail match) or ✗.  tail = tail match result.")
    print("  DC = confusion matrix cell (TP/FP/TN/FN).")
    print("=" * 100)
    hdr = ["Scenario", "Category", "Expected"] + model_keys
    print("  " + "  ".join(f"{h:<20}" for h in hdr))
    print("  " + "-" * (22 * len(hdr)))
    for scenario in SCENARIOS:
        exp_str = "NOMINAL" if scenario.expected_nominal else "REPLAN"
        row = [scenario.id, scenario.category[:14], exp_str]
        for mk in model_keys:
            recs = [r for r in all_results[mk] if r["scenario_id"] == scenario.id]
            if recs:
                r = recs[0]
                beh  = ("✓" if r["behavior_correct"] else "✗") if r["behavior_correct"] is not None else "?"
                dc   = r.get("decision_class", "?")
                tail = ""
                if r.get("tail_match") is not None:
                    tail = " tail:" + ("✓" if r["tail_match"] else "✗")
                row.append(f"{beh}[{dc}]{tail}")
            else:
                row.append("—")
        print("  " + "  ".join(f"{v:<20}" for v in row))

    # ------------------------------------------------------------------
    # CONFLICT_AND_ADAPT detail printout
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("EXPERIMENT 4 DETAIL — Conflict-and-Adapt Scenarios")
    print("  These scenarios require principled trade-offs with no explicit priority rules.")
    print("  CA1/CA2: partial mission (close cylinder mapped, far cylinder skipped).")
    print("  CA3: NOMINAL (uncertain detection, existing plan is already correct).")
    print("=" * 100)
    for mk in model_keys:
        ca_recs = [r for r in all_results[mk] if r["category"] == "conflict_and_adapt"]
        print(f"\n  {mk}:")
        for r in ca_recs:
            beh = ("✓" if r["behavior_correct"] else "✗") if r["behavior_correct"] is not None else "?"
            print(
                f"    [{r['scenario_id']}]  "
                f"expected={'NOMINAL' if r['expected_nominal'] else 'REPLAN':<7}  "
                f"produced={'NOMINAL' if r['nominal'] else 'REPLAN':<7}  "
                f"[{r.get('decision_class','?')}]  "
                f"behav:{beh}  "
                f"tail_states={r['tail_states']}"
            )
            if r.get("tail_match") is False:
                print(f"      tail_mismatch: {r['tail_match_detail']}")

    save_json(f"{output_dir}/exp4_results_raw.json", all_results)
    save_json(f"{output_dir}/exp4_summary.json", summary)
    print(f"\n  Saved → {output_dir}/")
    print("\nDone.\n")
    return all_results, summary


if __name__ == "__main__":
    MODELS_TO_TEST = ["gemini-flash-2.5", "qwen-235b", "llama-3.1-8b"]
    run_experiment(MODELS_TO_TEST, output_dir="exp4_output")