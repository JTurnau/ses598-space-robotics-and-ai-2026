#!/usr/bin/env python3
"""
exp1_planning_quality.py
========================
Experiment 1: Planning Quality

Evaluates how reliably each LLM can generate structurally executable and
semantically correct UAV mission plans from natural language.

Validity (structural executability only — is the plan runnable?):
  - First step is takeoff
  - Last step is return_home
  - approach always precedes map for the same cylinder_id
  - All state names are from the known vocabulary
  - mode values are "orbit" or "vertical_map" where applicable
  NOTE: continuous parameter values (standoff distances, altitudes) are NOT
  part of validity — they are evaluated under tail matching.

Tail matching (semantic correctness — does the plan do the right thing?):
  Step-by-step comparison of state, args keys, and repeat counts against a
  reference plan.  Continuous values that fall outside constraints count as
  tail mismatches and are captured in tail_match_pct.

Metrics collected per model:
  valid_pct         : % of missions where a structurally executable plan was produced
  tail_match_pct    : % of valid plans whose content matches the reference tail exactly
  perfect_tail_pct  : % of ALL missions with a structurally valid AND reference-matching plan
  constraint_sat_pct: % of step-level continuous constraints satisfied across all plans
  avg_attempts      : avg LLM calls before a valid plan (or max_attempts if never valid)
  avg_plan_length   : avg steps in valid plans
  avg_latency_s     : avg wall-clock seconds per mission (all attempts)

Mission set (15 missions, increasing complexity):
  Level 1 (Simple / no targets): 5 missions
  Level 2 (Single target):       5 missions
  Level 3 (Multi-target / complex): 5 missions

Output:
  exp1_results_raw.json  - full per-mission records for every model
  exp1_summary.json      - aggregate metrics table (models x metrics)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_utils import (
    MODELS, generate_plan, validate_plan, fmt_plan, save_json,
    MIN_STANDOFF_M, VMAP_MIN_STANDOFF_M, VMAP_MAX_STANDOFF_M, VMAP_MIN_ALTITUDE_M,
)

# ---------------------------------------------------------------------------
# STRUCTURAL VALIDITY  (executability only — no continuous value checks)
# ---------------------------------------------------------------------------

_VALID_STATES = frozenset({"takeoff", "search", "approach", "map", "return_home"})
_VALID_PATTERNS = frozenset({"yaw_scan", "lawnmower"})
_VALID_MODES = frozenset({"orbit", "vertical_map"})


def validate_plan_executable(plan: list) -> tuple[bool, list[str]]:
    """
    Check whether a plan is structurally executable.

    A plan is executable iff:
      1. It is a non-empty list of dicts.
      2. First step state is "takeoff".
      3. Last step state is "return_home".
      4. Every state name belongs to the known vocabulary.
      5. Every search step has a known pattern (yaw_scan | lawnmower) if
         a pattern arg is supplied.
      6. Every map step has a known mode (orbit | vertical_map) if a mode
         arg is supplied.
      7. Every map step is preceded (at some earlier position) by an approach
         step with the same cylinder_id.

    Continuous parameter values (standoff distances, altitudes, repeat counts)
    are intentionally NOT checked here — they are evaluated in tail matching.
    """
    errors: list[str] = []

    if not plan:
        return False, ["Plan is empty"]

    if not all(isinstance(s, dict) for s in plan):
        bad = [i for i, s in enumerate(plan) if not isinstance(s, dict)]
        errors.append(f"Non-dict steps at indices: {bad}")
        return False, errors

    # 1. Bookend checks
    if plan[0].get("state") != "takeoff":
        errors.append(f"First step must be takeoff, got '{plan[0].get('state')}'")
    if plan[-1].get("state") != "return_home":
        errors.append(f"Last step must be return_home, got '{plan[-1].get('state')}'")

    approached: set = set()

    for i, step in enumerate(plan):
        s    = step.get("state", "")
        args = step.get("args", {})

        # 2. Known state
        if s not in _VALID_STATES:
            errors.append(f"Step {i}: unknown state '{s}'")
            continue   # can't check further for this step

        # 3. search pattern — only flag if an explicit bad value is given
        if s == "search":
            pat = args.get("pattern")
            if pat is not None and pat not in _VALID_PATTERNS:
                errors.append(f"Step {i} (search): unknown pattern '{pat}'")

        # 4. approach — record cylinder_id for ordering check
        if s == "approach":
            cid = args.get("cylinder_id")
            approached.add(cid)

        # 5. map — mode validity + ordering
        if s == "map":
            mode = args.get("mode")
            if mode is not None and mode not in _VALID_MODES:
                errors.append(f"Step {i} (map): unknown mode '{mode}'")

            cid = args.get("cylinder_id")
            # "all" is always allowed without a specific approach
            if cid != "all" and cid not in approached:
                errors.append(
                    f"Step {i} (map, cylinder_id={cid}): "
                    f"no preceding approach for this cylinder_id"
                )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# TAIL MATCHING
# ---------------------------------------------------------------------------

_ARGS_KEYS = (
    "cylinder_id", "mode", "standoff_distance", "min_altitude_m",
    "altitude", "pattern",
)


def _steps_match(produced: dict, reference: dict) -> bool:
    """
    Return True if a produced step matches a reference step.

    Matching rules:
      - "state" must be identical.
      - "repeat" must be identical (defaults to 1 if absent).
      - For each args key present in the REFERENCE step, the produced step
        must carry the same value.  Extra keys in the produced step that are
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
    Compare a produced plan against the reference plan.

    Returns (match: bool, explanation: str).

    Special cases:
      - reference is None  → scenario is unscored (True, note).
      - produced is None   → always False.
    """
    if reference is None:
        return True, "No reference plan — mission is unscored."
    if produced is None:
        return False, "No valid plan produced; reference plan exists."
    if len(produced) != len(reference):
        return (
            False,
            f"Plan length mismatch: produced {len(produced)} steps, "
            f"reference has {len(reference)} steps.",
        )
    mismatches = []
    for i, (p, r) in enumerate(zip(produced, reference)):
        if not _steps_match(p, r):
            mismatches.append(f"  step[{i}]: produced {p!r} ≠ reference {r!r}")
    if mismatches:
        return False, "Step mismatch(es):\n" + "\n".join(mismatches)
    return True, "Plan matches reference exactly."


# ---------------------------------------------------------------------------
# CONSTRAINT SATISFACTION  (continuous parameter checks — feeds tail_match_pct)
# ---------------------------------------------------------------------------

def check_constraint_satisfaction(plan: list) -> tuple[int, int]:
    """
    Check step-level continuous parameter constraints.

    This captures the numeric quality of a plan that structural validity
    deliberately ignores: standoff distances, altitude bounds, etc.

    Returns (satisfied_count, total_checkable_count).
    """
    if not plan:
        return 0, 0

    sat   = 0
    total = 0

    prev_state = None

    for step in plan:
        if not isinstance(step, dict):
            continue
        s    = step.get("state")
        args = step.get("args", {})

        if s == "approach":
            # standoff_distance present and positive
            total += 1
            sd = args.get("standoff_distance")
            if sd is not None and float(sd) > 0:
                sat += 1

        if s == "map":
            mode = args.get("mode", "orbit")
            sd   = float(args.get("standoff_distance", 0))

            # ordering (approach immediately before map)
            total += 1
            if prev_state == "approach":
                sat += 1

            # standoff distance within mode-specific bounds
            total += 1
            if mode == "orbit" and sd >= MIN_STANDOFF_M:
                sat += 1
            elif mode == "vertical_map" and VMAP_MIN_STANDOFF_M <= sd <= VMAP_MAX_STANDOFF_M:
                sat += 1

            # repeat must be a top-level field, not inside args
            total += 1
            if "repeat" not in args:
                sat += 1

            # min_altitude_m check for vertical_map
            if mode == "vertical_map":
                total += 1
                min_alt = float(args.get("min_altitude_m", VMAP_MIN_ALTITUDE_M))
                if min_alt >= VMAP_MIN_ALTITUDE_M:
                    sat += 1

        if s == "takeoff":
            # altitude should be positive and present
            total += 1
            alt = args.get("altitude")
            if alt is not None and float(alt) > 0:
                sat += 1

        prev_state = s

    return sat, total


# ---------------------------------------------------------------------------
# STEP BUILDERS  (for reference tails)
# ---------------------------------------------------------------------------

def _step(state: str, args: dict | None = None, repeat: int = 1) -> dict:
    return {"state": state, "args": args or {}, "repeat": repeat}

def _takeoff(alt: float = 5.0) -> dict:
    return _step("takeoff", {"altitude": alt})

def _search(pattern: str = "yaw_scan") -> dict:
    return _step("search", {"pattern": pattern})

def _approach(cid, sd: float = 5.0) -> dict:
    return _step("approach", {"cylinder_id": cid, "standoff_distance": sd})

def _map_orbit(cid, sd: float = 5.0, repeat: int = 1) -> dict:
    return _step("map", {"cylinder_id": cid, "standoff_distance": sd, "mode": "orbit"}, repeat)

def _map_vmap(cid, sd: float = 5.0, repeat: int = 1) -> dict:
    return _step(
        "map",
        {"cylinder_id": cid, "standoff_distance": sd,
         "mode": "vertical_map", "min_altitude_m": 2.0},
        repeat,
    )

def _return_home() -> dict:
    return _step("return_home")


# ---------------------------------------------------------------------------
# MISSION SET  (unchanged)
# ---------------------------------------------------------------------------

MISSIONS: list[dict] = [
    # ---- Level 1: No target interaction ----
    {
        "id":         "L1-01",
        "level":      1,
        "mission":    "Take off and return home.",
        "note":       "Minimal plan, no search or mapping",
        "expected_tail": [
            _takeoff(5.0),
            _return_home(),
        ],
    },
    {
        "id":         "L1-02",
        "level":      1,
        "mission":    "Take off to 5 meters, do a quick scan of the area, then come back.",
        "note":       "yaw_scan expected given 'quick'",
        "expected_tail": [
            _takeoff(5.0),
            _search("yaw_scan"),
            _return_home(),
        ],
    },
    {
        "id":         "L1-03",
        "level":      1,
        "mission":    "Perform a thorough search of the field and return to base.",
        "note":       "lawnmower expected given 'thorough'",
        "expected_tail": [
            _step("takeoff", {}),   # any altitude
            _search("lawnmower"),
            _return_home(),
        ],
    },
    {
        "id":         "L1-04",
        "level":      1,
        "mission":    "Fly up to 3 meters, sweep the area for anything interesting, land at home.",
        "note":       "Low altitude, indoor-ish context",
        "expected_tail": [
            _takeoff(3.0),
            _search("yaw_scan"),
            _return_home(),
        ],
    },
    {
        "id":         "L1-05",
        "level":      1,
        "mission":    "Conduct a systematic grid search of the exploration zone and return when done.",
        "note":       "lawnmower strongly implied",
        "expected_tail": [
            _step("takeoff", {}),
            _search("lawnmower"),
            _return_home(),
        ],
    },

    # ---- Level 2: Single target ----
    {
        "id":         "L2-01",
        "level":      2,
        "mission":    "Search for cylinders and map the first one you find, then come home.",
        "note":       "Single target, orbit mode default",
        "expected_tail": [
            _step("takeoff", {}),
            _search("yaw_scan"),
            _approach(1),
            _map_orbit(1),
            _return_home(),
        ],
    },
    {
        "id":         "L2-02",
        "level":      2,
        "mission":    "Find cylinders and do a vertical mapping sweep of each one.",
        "note":       "vertical_map mode, all cylinders",
        "expected_tail": [
            _step("takeoff", {}),
            _step("search",  {}),
            _approach("all"),
            _map_vmap("all"),
            _return_home(),
        ],
    },
    {
        "id":         "L2-03",
        "level":      2,
        "mission":    "Scan for objects. If you find any, orbit the first one twice, then return.",
        "note":       "orbit x2 (repeat:2), first cylinder only",
        "expected_tail": [
            _step("takeoff", {}),
            _search("yaw_scan"),
            _approach(1),
            _map_orbit(1, repeat=2),
            _return_home(),
        ],
    },
    {
        "id":         "L2-04",
        "level":      2,
        "mission":    "Take off to 6 meters, do a panoramic scan, approach any cylinder at 5 meters and orbit it once.",
        "note":       "Explicit altitude and standoff",
        "expected_tail": [
            _takeoff(6.0),
            _search("yaw_scan"),
            _approach("all", sd=5.0),
            _map_orbit("all", sd=5.0, repeat=1),
            _return_home(),
        ],
    },
    {
        "id":         "L2-05",
        "level":      2,
        "mission":    "Search the area thoroughly, then do two full vertical mapping laps on each cylinder you find.",
        "note":       "lawnmower, vertical_map, repeat:2",
        "expected_tail": [
            _step("takeoff", {}),
            _search("lawnmower"),
            _approach("all"),
            _map_vmap("all", repeat=2),
            _return_home(),
        ],
    },

    # ---- Level 3: Multi-target / complex ----
    {
        "id":         "L3-01",
        "level":      3,
        "mission":    "Search for cylinders and map every one you find with a single orbit, then return.",
        "note":       "All cylinders, orbit x1",
        "expected_tail": [
            _step("takeoff", {}),
            _step("search",  {}),
            _approach("all"),
            _map_orbit("all", repeat=1),
            _return_home(),
        ],
    },
    {
        "id":         "L3-02",
        "level":      3,
        "mission":    "Find any cylinders and map each one 2 times, then come home.",
        "note":       "All cylinders, repeat:2",
        "expected_tail": [
            _step("takeoff", {}),
            _step("search",  {}),
            _approach("all"),
            _map_orbit("all", repeat=2),
            _return_home(),
        ],
    },
    {
        "id":         "L3-03",
        "level":      3,
        "mission":    (
            "Perform a comprehensive survey: take off to 8 meters, "
            "do a lawnmower search, then do a vertical mapping sweep on every "
            "cylinder found, and return to launch."
        ),
        "note":       "lawnmower + vertical_map, all",
        "expected_tail": [
            _takeoff(8.0),
            _search("lawnmower"),
            _approach("all"),
            _map_vmap("all"),
            _return_home(),
        ],
    },
    {
        "id":         "L3-04",
        "level":      3,
        "mission":    (
            "Take off, scan quickly, then orbit any cylinders you find "
            "three times each to build a detailed map. Come home afterwards."
        ),
        "note":       "yaw_scan + orbit x3, all",
        "expected_tail": [
            _step("takeoff", {}),
            _search("yaw_scan"),
            _approach("all"),
            _map_orbit("all", repeat=3),
            _return_home(),
        ],
    },
    {
        "id":         "L3-05",
        "level":      3,
        "mission":    (
            "Survey the area using a grid search pattern. For each cylinder "
            "discovered, do two vertical mapping laps at a close 4 meter standoff. "
            "Return home when complete."
        ),
        "note":       "lawnmower + vertical_map x2 at 4m standoff, all",
        "expected_tail": [
            _step("takeoff", {}),
            _search("lawnmower"),
            _approach("all", sd=4.0),
            _map_vmap("all", sd=4.0, repeat=2),
            _return_home(),
        ],
    },
]

# Fix L1-01, L1-03, L1-05 to use unconstrained takeoff altitude
# (reference omits the altitude key so any reasonable value matches)
for _m in MISSIONS:
    if _m["id"] in ("L1-01", "L1-03", "L1-05"):
        tail = _m["expected_tail"]
        tail[0] = _step("takeoff", {})


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(model_keys: list[str], output_dir: str = "."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list] = {mk: [] for mk in model_keys}

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: PLANNING QUALITY")
    print("=" * 70)
    print()
    print("Metrics:")
    print("  valid%        = % missions producing a structurally executable plan")
    print("                  (correct ordering, known state/mode names — no numeric checks)")
    print("  tail%         = % of valid plans matching the reference tail exactly")
    print("                  (state, args incl. continuous values, repeat counts)")
    print("  perfect_tail% = % of ALL missions with a valid + reference-matching plan [PRIMARY]")
    print("  CS%           = % of step-level continuous constraints satisfied")
    print()

    for mission_cfg in MISSIONS:
        mid      = mission_cfg["id"]
        mission  = mission_cfg["mission"]
        level    = mission_cfg["level"]
        note     = mission_cfg["note"]
        ref_tail = mission_cfg["expected_tail"]

        print(f"\n--- [{mid}] Level {level}: {mission[:60]}...")

        for model_key in model_keys:
            print(f"  Model: {model_key}")

            # generate_plan uses the original validate_plan from experiment_utils;
            # we re-validate here with our lighter executability check so the
            # loop can still benefit from the multi-attempt wrapper, but we
            # record our own validity flag for metrics.
            result = generate_plan(mission, model_key)

            # Re-assess validity using executability-only check
            if result["plan"] is not None:
                exec_valid, exec_errors = validate_plan_executable(result["plan"])
            else:
                exec_valid  = False
                exec_errors = ["No plan produced"]

            # Tail match (only meaningful if executably valid)
            if exec_valid and result["plan"] is not None:
                tail_match, tail_match_detail = _tails_match(result["plan"], ref_tail)
            else:
                tail_match        = False
                tail_match_detail = "Plan not executable; tail match skipped."

            # Constraint satisfaction (continuous params)
            sat, total = check_constraint_satisfaction(result["plan"])

            # Perfect tail = executable AND matches reference
            perfect_tail = exec_valid and tail_match

            result["mission_id"]        = mid
            result["mission_level"]     = level
            result["exec_valid"]        = exec_valid
            result["exec_errors"]       = exec_errors
            result["tail_match"]        = tail_match
            result["tail_match_detail"] = tail_match_detail
            result["perfect_tail"]      = perfect_tail
            result["constraint_sat"]    = sat
            result["constraint_total"]  = total
            result["reference_tail"]    = ref_tail

            val_mark  = "✓" if exec_valid  else "✗"
            tail_mark = "✓" if tail_match  else "✗"
            cs_str    = f"{sat}/{total}" if total > 0 else "n/a"
            print(
                f"    valid:{val_mark}  tail:{tail_mark}  "
                f"constraints={cs_str}  attempts={result['attempts']}  "
                f"latency={result['latency_s']:.1f}s"
            )
            if exec_valid and not tail_match:
                print(f"      tail_mismatch: {tail_match_detail}")
                ref_states  = [s.get("state") for s in ref_tail]
                prod_states = [s.get("state") for s in result["plan"]]
                print(f"      reference: {ref_states}")
                print(f"      produced:  {prod_states}")
            if not exec_valid:
                for e in exec_errors[:3]:
                    print(f"      invalid: {e}")

            all_results[model_key].append(result)

    # ------------------------------------------------------------------
    # Aggregate summary — overall + broken down by level
    # ------------------------------------------------------------------
    summary: dict[str, dict] = {}
    levels = [1, 2, 3]

    def _agg(recs: list[dict]) -> dict:
        nr = len(recs)
        if nr == 0:
            return {}

        n_valid       = sum(1 for r in recs if r["exec_valid"])
        n_tail_ok     = sum(1 for r in recs if r["tail_match"])
        n_perfect     = sum(1 for r in recs if r["perfect_tail"])
        total_sat     = sum(r["constraint_sat"]   for r in recs)
        total_chk     = sum(r["constraint_total"] for r in recs)
        avg_attempts  = sum(r["attempts"]         for r in recs) / nr
        plan_lengths  = [len(r["plan"]) for r in recs if r["plan"]]
        avg_plan_len  = sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0.0
        avg_latency   = sum(r["latency_s"]        for r in recs) / nr

        # tail_match_pct denominator = valid plans only
        tail_denom = max(n_valid, 1)

        return {
            "n_missions":        nr,
            # % missions with an executable plan
            "valid_pct":         round(100 * n_valid   / nr,         1),
            # % of executable plans that matched the reference exactly
            "tail_match_pct":    round(100 * n_tail_ok / tail_denom, 1),
            # % of ALL missions with executable + reference-matching plan [PRIMARY]
            "perfect_tail_pct":  round(100 * n_perfect / nr,         1),
            # continuous parameter constraint satisfaction
            "constraint_sat_pct": round(100 * total_sat / max(total_chk, 1), 1),
            "avg_attempts":      round(avg_attempts, 2),
            "avg_plan_length":   round(avg_plan_len,  1),
            "avg_latency_s":     round(avg_latency,   2),
        }

    for model_key in model_keys:
        records  = all_results[model_key]
        by_level = {lv: _agg([r for r in records if r["mission_level"] == lv])
                    for lv in levels}
        summary[model_key] = {"overall": _agg(records), "by_level": by_level}

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    cols = ["valid%", "tail%", "perfect%", "CS%", "attempts", "planLen", "latency"]

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY — Table 1: Overall Results")
    print()
    print("  valid%   = % missions producing a structurally executable plan")
    print("             (bookend ordering, known state/mode vocab; no numeric checks)")
    print()
    print("  tail%    = % of valid plans matching reference step-by-step")
    print("             (state, args incl. continuous values, repeat counts)")
    print("             denominator = valid plans only")
    print()
    print("  perfect% = % of ALL missions with valid + reference-matching plan [PRIMARY]")
    print("             perfect% = valid% × tail%/100")
    print()
    print("  CS%      = % of individual continuous parameter constraints satisfied")
    print("             (standoff bounds, altitude bounds, repeat placement)")
    print("=" * 80)
    print(f"{'Model':<22} " + "  ".join(f"{c:>9}" for c in cols))
    print("-" * 90)
    for mk, s in summary.items():
        o = s["overall"]
        print(
            f"{mk:<22}  "
            f"{o.get('valid_pct',          0):>9.1f}  "
            f"{o.get('tail_match_pct',     0):>9.1f}  "
            f"{o.get('perfect_tail_pct',   0):>9.1f}  "
            f"{o.get('constraint_sat_pct', 0):>9.1f}  "
            f"{o.get('avg_attempts',       0):>9.2f}  "
            f"{o.get('avg_plan_length',    0):>9.1f}  "
            f"{o.get('avg_latency_s',      0):>8.2f}s"
        )

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY — Table 2: Breakdown by Complexity Level")
    print("  Level 1: No target interaction  |  Level 2: Single target  |  Level 3: Multi-target")
    print("=" * 80)
    for lv in levels:
        print(f"\n  [LEVEL {lv}]")
        print(f"  {'Model':<22} " + "  ".join(f"{c:>9}" for c in cols))
        print("  " + "-" * 88)
        for mk, s in summary.items():
            b = s["by_level"].get(lv, {})
            if not b:
                print(f"  {mk:<22}  (no records)")
                continue
            print(
                f"  {mk:<22}  "
                f"{b.get('valid_pct',          0):>9.1f}  "
                f"{b.get('tail_match_pct',     0):>9.1f}  "
                f"{b.get('perfect_tail_pct',   0):>9.1f}  "
                f"{b.get('constraint_sat_pct', 0):>9.1f}  "
                f"{b.get('avg_attempts',       0):>9.2f}  "
                f"{b.get('avg_plan_length',    0):>9.1f}  "
                f"{b.get('avg_latency_s',      0):>8.2f}s"
            )

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY — Table 3: Per-Mission Outcome Matrix")
    print("  valid = executable plan produced.  tail = reference match.  perfect = both.")
    print("=" * 80)
    hdr = ["Mission", "Level", "Note"] + model_keys
    print("  " + "  ".join(f"{h:<20}" for h in hdr))
    print("  " + "-" * (22 * len(hdr)))
    for mission_cfg in MISSIONS:
        mid   = mission_cfg["id"]
        level = mission_cfg["level"]
        note  = mission_cfg["note"][:14]
        row   = [mid, str(level), note]
        for mk in model_keys:
            recs = [r for r in all_results[mk] if r["mission_id"] == mid]
            if recs:
                r   = recs[0]
                v   = "✓" if r["exec_valid"]  else "✗"
                t   = "✓" if r["tail_match"]  else "✗"
                p   = "✓" if r["perfect_tail"] else "✗"
                row.append(f"v{v} t{t} p{p}")
            else:
                row.append("—")
        print("  " + "  ".join(f"{v:<20}" for v in row))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_json(f"{output_dir}/exp1_results_raw.json", all_results)
    save_json(f"{output_dir}/exp1_summary.json", summary)
    print(f"\n  Saved → {output_dir}/")
    print("\nDone.\n")
    return all_results, summary


if __name__ == "__main__":
    MODELS_TO_TEST = [
        "gemini-flash-2.5",
        "qwen-235b",
        "llama-3.1-8b",
    ]
    run_experiment(MODELS_TO_TEST, output_dir="exp1_output")