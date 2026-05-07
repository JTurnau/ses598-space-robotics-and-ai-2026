#!/usr/bin/env python3
"""

Plan shape
----------
A plan is a list of step dicts:
  { "state": <skill_name>, "args": { ... }, "repeat": <int, default 1> }

repeat: N causes the executor to run that skill to completion N times before
advancing - this is how "map 3 times" is expressed natively.

LLM backend
-----------
Set LLM_BACKEND to select the inference provider:

  LLM_BACKEND = "gemini"    # Google Gemini via google-genai
                             #   pip install google-genai
                             #   set GEMINI_API_KEY or fill GEMINI_API_KEY below

  LLM_BACKEND = "cerebras"  # Cerebras via cerebras-cloud-sdk (original)
                             #   pip install cerebras-cloud-sdk
                             #   set CEREBRAS_API_KEY or fill below
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                        QoSProfile, QoSReliabilityPolicy)

from geometry_msgs.msg import Point
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                           VehicleCommand, VehicleOdometry, VehicleStatus)
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray


# ---------------------------------------------------------------------------
# LLM BACKEND CONFIG
# ---------------------------------------------------------------------------
LLM_BACKEND = "gemini"   # <-- change to "cerebras" to revert to cerebras models

import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
GEMINI_MODEL   = "gemini-2.5-flash"

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "YOUR_CEREBRAS_KEY_HERE")
CEREBRAS_MODEL   = "llama3.1-8b"

MIN_STANDOFF_M: float = 5.0

# Vertical map standoff constraints (separate, more flexible than orbit)
VMAP_MIN_STANDOFF_M: float = 3.0
VMAP_MAX_STANDOFF_M: float = 7.0

# Minimum safe AGL the drone is allowed to descend to during vertical_map
VMAP_MIN_ALTITUDE_M: float = 2.0

_NO_REPLAN_AFTER: frozenset[str] = frozenset({"takeoff", "return_home"})

# ---------------------------------------------------------------------------
# LLM CLIENT INITIALIZATION
# ---------------------------------------------------------------------------

if LLM_BACKEND == "gemini":
    from google import genai as _genai
    from google.genai import types as _genai_types
    _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
elif LLM_BACKEND == "cerebras":
    from cerebras.cloud.sdk import Cerebras as _Cerebras
    _cerebras_client = _Cerebras(api_key=CEREBRAS_API_KEY)
else:
    raise ValueError(f"Unknown LLM_BACKEND={LLM_BACKEND!r}. Use 'gemini' or 'cerebras'.")


# ---------------------------------------------------------------------------
# UNIFIED query_llm()
# ---------------------------------------------------------------------------

def query_llm(prompt: str, system: str, max_tokens: int = 4096) -> str:
    if LLM_BACKEND == "gemini":
        config = _genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
        )
        for attempt in range(3):
            try:
                response = _gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt if prompt else "Respond now.",
                    config=config,
                )
                return response.text
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    wait = 30 * (attempt + 1)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[LLM/gemini] Rate limited - waiting {wait}s "
                        f"(attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("query_llm (gemini): failed after 3 retries")
    else:
        resp = _cerebras_client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=0,
            stream=False,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------

PLANNING_PROMPT = """
You are a UAV mission planner.
Given a natural language mission description, produce a JSON execution plan
as a sequence of parameterised skill invocations chosen from the vocabulary below.
The UAV operates in an unknown environment - objects are discovered only through
onboard perception during flight, not known in advance.

=== AVAILABLE SKILLS ===

Each plan step has the shape:
  { "state": <skill_name>, "args": { ... }, "repeat": <int, default 1> }

"repeat": N runs that skill to completion N times before advancing.

SKILL          ARGS                                   NOTES
-------------------------------------------------------------------------------
takeoff        altitude: float (m)                    ALWAYS the first step.
                                                      Infer a reasonable altitude
                                                      from context if not stated:
                                                      confined space  ->  2-3 m
                                                      open area       ->  5-8 m

search         pattern: "yaw_scan"                    Discovers objects via onboard
                       | "lawnmower"                  perception during flight.
                                                      yaw_scan  - rotate in place;
                                                                  good for small areas
                                                                  or quick sweeps.
                                                      lawnmower - systematic grid
                                                                  coverage; use when
                                                                  thorough search of a
                                                                  large area is needed.

approach       cylinder_id: int | "all"               Fly to standoff_distance from
               standoff_distance: float (m, min 5.0)  the specified target(s).
                                                      MUST appear before map for the
                                                      same cylinder_id.

map            cylinder_id: int | "all"               Two mapping modes available:
               standoff_distance: float (m)
               mode: "orbit"                          ORBIT MODE (mode: "orbit"):
                    | "vertical_map"                    Orbit the target once per
                                                        invocation. standoff >= 5.0 m.
                                                        Use repeat: N to orbit N times.
                                                        approach for the same
                                                        cylinder_id MUST precede this.

                                                      VERTICAL MAP MODE (mode: "vertical_map"):
                                                        Performs a systematic vertical
                                                        sweep around the cylinder in
                                                        four 90-degree arc segments.
                                                        Each segment: centre on cylinder,
                                                        descend to min_altitude_m (>=2.0m),
                                                        ascend above cylinder top, then
                                                        quarter-orbit. standoff 3.0?7.0 m.
                                                        Use repeat: N for N full laps.
                                                        approach for the same
                                                        cylinder_id MUST precede this.

               min_altitude_m: float (m, >= 2.0)      Only for vertical_map mode.
                                                       Default 2.0. Minimum AGL the
                                                       drone descends to during sweep.

return_home    (no args)                              Always the last step.
                                                      Triggers autonomous return-to-launch.

=== CONSTRAINTS ===

  1. First step MUST be takeoff. Last step must be return_home.
  2. approach(cylinder_id=X) MUST appear immediately before map(cylinder_id=X).
  3. Do not include approach or map unless the mission explicitly involves object
     interaction.
  4. Do not add steps that are not implied by the mission description.
  5. For mode "orbit": standoff_distance must be >= 5.0 m for both approach and map.
  6. For mode "vertical_map": standoff_distance must be between 3.0 m and 7.0 m.
  7. After search, assume objects may be found. The automatic replanner will
     adjust the plan based on what is actually discovered - you do not need to
     handle "nothing found" cases explicitly.

=== OUTPUT ===

Respond with a JSON array only. No explanation, no markdown, no backticks.
"""


# ---------------------------------------------------------------------------
# REPLAN PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_replan_system_prompt(
    mission_intent: str,
    completed_steps: list[dict],
    remaining_steps: list[dict],
    cylinders: list,
) -> str:
    def _fmt_steps(steps: list[dict], annotate_semantics: bool = False) -> str:
        if not steps:
            return "  (none)"
        lines = []
        for i, st in enumerate(steps):
            args_str = ", ".join(f"{k}={v}" for k, v in st.get("args", {}).items())
            rep      = st.get("repeat", 1)
            rep_str  = f" x{rep}" if rep > 1 else ""
            line     = f"  [{i}] {st['state']:22} {args_str}{rep_str}"
            if annotate_semantics:
                if st["state"] == "approach":
                    line += "  # repositioning only - NO mapping performed"
                elif st["state"] == "map":
                    mode = st.get("args", {}).get("mode", "orbit")
                    line += f"  # {rep} {mode} pass(es) COMPLETED"
            lines.append(line)
        return "\n".join(lines)

    def _build_manifest(cyls: list) -> str:
        if not cyls:
            return "  No cylinders discovered yet."
        data = [
            {
                "cylinder_id":          c.id,
                "world_ned_x_m":        round(c.world_x, 2),
                "world_ned_y_m":        round(c.world_y, 2),
                "depth_at_detection_m": round(c.depth_m, 2),
            }
            for c in cyls
        ]
        return (
            f"  {len(cyls)} cylinder(s) discovered:\n"
            "  ```json\n"
            + "  " + json.dumps(data, indent=2).replace("\n", "\n  ")
            + "\n  ```"
        )

    mapped = _extract_mapped_cylinders(completed_steps)

    if mapped:
        mapped_lines = [
            f"  - Cylinder {cid}: COMPLETE ({passes} pass(es) done)"
            for cid, passes in sorted(mapped.items())
        ]
        mapped_section = "\n".join(mapped_lines)
    else:
        mapped_section = "  (none yet)"

    unmapped_ids = [c.id for c in cylinders if c.id not in mapped]
    if unmapped_ids:
        unmapped_note = (
            "  Cylinders NOT yet mapped: "
            + ", ".join(str(i) for i in unmapped_ids)
        )
    else:
        unmapped_note = "  All discovered cylinders have been mapped."

    return f"""\
=== SECTION 1 - BACKGROUND ===

You are a mid-flight autonomous UAV mission replanner.

After each skill completes, you receive a status snapshot of the ongoing
mission. Your role is to review the remaining plan and decide whether it
still satisfies the user's original intent given what has been discovered
so far. You may revise, extend, or trim the remaining steps as needed.
You do NOT re-plan from scratch - only the tail (steps yet to execute)
is yours to change. Completed steps are fixed history. Do NOT make changes
unless they are ABSOLUTELY NECESSARY to complete the intended mission.


=== SECTION 2 - USER MISSION ===

  "{mission_intent}"

This is the exact mission the user requested. Use it as the ground truth
for what "success" means. Every decision you make should serve this intent.


=== SECTION 3 - AVAILABLE SKILLS ===

Only the following skills may appear in a revised tail plan.
takeoff and search are already complete - do NOT include them.

SKILL         ARGS                                   NOTES
------------------------------------------------------------------------------
approach      cylinder_id: int                       Fly to standoff_distance
              standoff_distance: float (m)           from a specific cylinder.
                                                     MUST immediately precede
                                                     the map step for the same
                                                     cylinder_id.
                                                     For orbit: standoff >= 5.0 m
                                                     For vertical_map: 3.0?7.0 m

map           cylinder_id: int                       Two modes:
              standoff_distance: float (m)
              mode: "orbit"                          ORBIT: one full orbit per
                   | "vertical_map"                  invocation (standoff >= 5.0 m).
                                                     Use repeat: N to orbit N times.

              min_altitude_m: float (>= 2.0)         VERTICAL_MAP: systematic vertical
                                                     sweep in four 90-deg arc segments
                                                     (standoff 3.0?7.0 m). Use repeat: N
                                                     for N full laps. min_altitude_m
                                                     sets the lowest AGL point (>= 2.0m,
                                                     default 2.0).

return_home   (no args)                              Must be the last step.

Note that argument 'all' appearing in previous plan is a placeholder for mapping
cylinders that should be filled in with the pattern described below.

Required pattern for each cylinder you intend to map:
  {{"state":"approach","args":{{"standoff_distance":D,"cylinder_id":N}}}},
  {{"state":"map",     "args":{{"mode":"orbit","standoff_distance":D,"cylinder_id":N}},"repeat":K}}

  OR for vertical_map mode:
  {{"state":"approach","args":{{"standoff_distance":D,"cylinder_id":N}}}},
  {{"state":"map",     "args":{{"mode":"vertical_map","standoff_distance":D,"cylinder_id":N,"min_altitude_m":2.0}},"repeat":K}}

Hard constraints:
  - For orbit: standoff_distance >= 5.0 for both approach and map
  - For vertical_map: 3.0 <= standoff_distance <= 7.0 for both approach and map
  - min_altitude_m >= 2.0 (vertical_map only)
  - approach(cylinder_id=X) MUST immediately precede map(cylinder_id=X)
  - "repeat" is a TOP-LEVEL step field, never inside args
  - Last step MUST be return_home
  - Do NOT add approach or map for already-mapped cylinders


=== SECTION 4 - MISSION STATUS REPORT ===

COMPLETED STEPS (fixed history):
{_fmt_steps(completed_steps, annotate_semantics=True)}

MAPPING PROGRESS (ground truth - authoritative, derived from execution history):
    IMPORTANT: "approach" steps only reposition the drone. They do NOT count as mapping.
    Only completed "map" steps count as mapping passes.
{mapped_section}

{unmapped_note}

REMAINING PLAN (scheduled for execution in this order):
{_fmt_steps(remaining_steps)}

DISCOVERED OBJECTS:
{_build_manifest(cylinders)}


=== SECTION 5 - OUTPUT DIRECTIVE ===

Do NOT repeat, echo, or summarise any part of this prompt in your response.

Carefully compare the remaining plan against the user mission.
If the remaining plan already fulfils the user mission, respond
with exactly this single token and nothing else
(no preamble, no explanation, no repetition of these instructions):

NOMINAL

ONLY if the remaining plan MUST be adjusted, output ONLY:
  1. A corrected JSON array of the complete revised tail (all steps from
     now until return_home). Every element must be a complete JSON object.
     Example (orbit):
     [{{"state":"approach","args":{{"standoff_distance":5.0,"cylinder_id":2}}}},
     {{"state":"map","args":{{"mode":"orbit","standoff_distance":5.0,"cylinder_id":2}},"repeat":2}},
     {{"state":"return_home","args":{{}}}}]

     Example (vertical_map):
     [{{"state":"approach","args":{{"standoff_distance":5.0,"cylinder_id":2}}}},
     {{"state":"map","args":{{"mode":"vertical_map","standoff_distance":5.0,"cylinder_id":2,"min_altitude_m":2.0}},"repeat":1}},
     {{"state":"return_home","args":{{}}}}]

  2. Immediately after the JSON array, on a new line starting with
     "REASON:", a single concise sentence explaining what was wrong with
     the old plan and why your revision is necessary to complete the
     user mission.

Do not include any other explanation, markdown, or backticks.
"""


# ---------------------------------------------------------------------------
# MISSIONS  (edit to test different natural language inputs)
# ---------------------------------------------------------------------------

MISSIONS = [
    # "Quickly scan the area, then come home.",
    # "Search for cylinders and map every one you find.",
    # "Find any cylinders and map each one 2 times.",
    # "Approach and inspect the first cylinder you see, then return.",
    "Find cylinders and do a vertical map of each one.",
    # "Search for cylinders and do 2 vertical mapping laps of each one you find.",
]


# ---------------------------------------------------------------------------
# LLM HELPERS
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list:
    """
    Parse a JSON array from LLM output, repairing common model defects.
    Returns a list containing only dict elements.
    """
    text = re.sub(r"```[\w]*", "", text).strip("`").strip()
    for bad, good in {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }.items():
        text = text.replace(bad, good)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    text = re.sub(r'"args":\s*"\{\}"', '"args": {}', text)

    text = re.sub(r'(?<=\[)\s*"state"\s*:', '{"state":', text)
    text = re.sub(r'(?<=\])\s*,\s*"state"\s*:', ',{"state":', text)
    text = re.sub(r'(?<=\})\s*,\s*"state"\s*:', ',{"state":', text)

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    clean = [item for item in parsed if isinstance(item, dict)]
    if len(clean) != len(parsed):
        dropped = [item for item in parsed if not isinstance(item, dict)]
        import logging
        logging.getLogger(__name__).warning(
            f"extract_json: dropped {len(dropped)} non-dict element(s): {dropped}")
    return clean


def validate_plan(plan: list, is_tail: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not plan:
        return False, ["Plan is empty"]

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            errors.append(f"Step {i} is not a dict (got {type(step).__name__}: {step!r})")
    if errors:
        return False, errors

    if not is_tail and plan[0]["state"] != "takeoff":
        errors.append("Must start with takeoff")
    if plan[-1]["state"] != "return_home":
        errors.append("Must end with return_home")

    valid_states = {"takeoff", "search", "approach", "map", "return_home"}
    approached: set[Any] = set()

    for i, step in enumerate(plan):
        s, args = step["state"], step.get("args", {})
        if s not in valid_states:
            errors.append(f"Unknown state '{s}' at step {i}")
            continue
        repeat = step.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1:
            errors.append(f"Invalid repeat={repeat!r} at step {i}")

        if s == "takeoff" and "altitude" not in args:
            errors.append(f"takeoff at {i} missing altitude")
        if s == "search" and args.get("pattern") not in ("yaw_scan", "lawnmower"):
            errors.append(f"search at {i} invalid pattern: {args.get('pattern')}")
        if s == "approach":
            sd  = args.get("standoff_distance", 0)
            cid = args.get("cylinder_id")
            if "standoff_distance" not in args:
                errors.append(f"approach at {i} missing standoff_distance")
            approached.add(cid)
        if s == "map":
            cid  = args.get("cylinder_id")
            sd   = args.get("standoff_distance", 0)
            mode = args.get("mode", "orbit")

            if cid not in approached and cid != "all":
                errors.append(
                    f"map at {i} (cylinder_id={cid}) requires a preceding "
                    f"approach for the same cylinder_id")

            if mode not in ("orbit", "vertical_map"):
                errors.append(f"map at {i} invalid mode={mode!r} (must be 'orbit' or 'vertical_map')")

            if mode == "orbit":
                if float(sd) < MIN_STANDOFF_M:
                    errors.append(
                        f"map(orbit) at {i} standoff_distance={sd} < minimum {MIN_STANDOFF_M}m")
            elif mode == "vertical_map":
                if float(sd) < VMAP_MIN_STANDOFF_M:
                    errors.append(
                        f"map(vertical_map) at {i} standoff_distance={sd} < minimum {VMAP_MIN_STANDOFF_M}m")
                if float(sd) > VMAP_MAX_STANDOFF_M:
                    errors.append(
                        f"map(vertical_map) at {i} standoff_distance={sd} > maximum {VMAP_MAX_STANDOFF_M}m")
                min_alt = args.get("min_altitude_m", VMAP_MIN_ALTITUDE_M)
                if float(min_alt) < VMAP_MIN_ALTITUDE_M:
                    errors.append(
                        f"map(vertical_map) at {i} min_altitude_m={min_alt} < safety floor {VMAP_MIN_ALTITUDE_M}m")

            if "repeat" in args:
                errors.append(
                    f"map at {i} has 'repeat' inside args - it must be a "
                    f"top-level step field")

    return len(errors) == 0, errors


def print_plan(plan: list, logger=None, current_step: int = -1, label: str = ""):
    emit  = logger.info if logger else print
    width = 62

    if label:
        emit("=" * width)
        emit(f"  {label}")
        emit("=" * width)

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            emit(f"  ??? [{i}] <invalid step: {step!r}>")
            continue
        args_str = ", ".join(f"{k}={v}" for k, v in step.get("args", {}).items())
        rep      = step.get("repeat", 1)
        rep_str  = f" x{rep}" if rep > 1 else ""

        if current_step < 0:
            marker = "   "
        elif i < current_step:
            marker = "[v]"
        elif i == current_step:
            marker = ">>>"
        else:
            marker = "   "

        emit(f"  {marker} [{i}] {step['state']:22} {args_str}{rep_str}")

    if label:
        emit("=" * width)


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class SpottedCylinder:
    id:       int
    ned_x:    float
    ned_y:    float
    ned_z:    float
    yaw:      float
    depth_m:  float
    world_x:  float
    world_y:  float
    px_cx:    float
    px_cy:    float
    width_px: float


@dataclass
class ExecutionContext:
    """Shared state passed to every skill tick."""
    cylinders: list[SpottedCylinder] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SKILL BASE
# ---------------------------------------------------------------------------

class Skill:
    def tick(self, args: dict, ctx: ExecutionContext, node: "MissionExecutorNode") -> bool:
        raise NotImplementedError

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# SKILLS
# ---------------------------------------------------------------------------

class TakeoffSkill(Skill):
    def tick(self, args, ctx, node):
        altitude = float(args["altitude"])
        cx, cy, _ = node.current_pos()
        node.publish_trajectory_setpoint(x=cx, y=cy, z=-altitude, yaw=0.0)
        if node.tick_count % 10 == 0:
            node.get_logger().info(
                f"[TAKEOFF] target={altitude:.1f}m  "
                f"current={node.current_altitude():.2f}m")
        return abs(node.current_altitude() - altitude) < node.POSITION_THRESHOLD


class SearchYawScanSkill(Skill):
    def __init__(self):
        self._state: dict = {}

    def reset(self):
        self._state = {}

    def tick(self, args, ctx, node):
        s = self._state

        if "start_yaw" not in s:
            s["start_yaw"]   = node.current_yaw()
            s["accumulated"] = 0.0
            node._publish_search_active(True)
            node.get_logger().info("[SEARCH/yaw_scan] Starting 360 deg scan - gate OPEN")

        cx, cy, cz = node.current_pos()
        yaw_speed        = 0.05
        s["start_yaw"]   += yaw_speed
        s["accumulated"] += yaw_speed
        node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["start_yaw"])

        if s["accumulated"] >= 2 * math.pi:
            node._publish_search_active(False)
            ctx.cylinders = list(node.spotted)
            node.get_logger().info(
                f"[SEARCH/yaw_scan] Complete - gate CLOSED.  "
                f"Spotted {len(ctx.cylinders)} cylinder(s).")
            return True
        return False


class ApproachSkill(Skill):
    """
    Flies to standoff_distance from one or more cylinders.

    Phase pipeline per target:
      A - Coarse fly to a position standoff_distance behind the cylinder.
      S - Short settle so perception stabilises.
      B - Fine yaw adjustment: rotate until the cylinder is centred in frame.
      C - Advance/retreat along the facing ray until live depth == standoff.
      R - Recovery: fly back to the cylinder's NED position and restart S.
    """

    CENTER_PX_TOLERANCE:    float = 20.0
    CENTER_CONFIRM_TICKS:   int   = 15
    CENTER_YAW_STEP_MAX:    float = 0.005
    CENTER_YAW_GAIN:        float = 0.00005
    CENTER_ADJUST_SETTLE_S: float = 0.6
    APPROACH_SETTLE_S:      float = 3.0
    INITIAL_SETTLE_S:       float = 4.0
    LOST_SIGHT_TIMEOUT_S:   float = 1.5
    MAX_RETRIES:            int   = 3
    DEPTH_STEP_M:           float = 0.3

    def __init__(self):
        self._s: dict = {}

    def reset(self):
        self._s = {}

    def _resolve_targets(self, args, ctx, node):
        cid = args.get("cylinder_id")
        if cid is None or cid == "all":
            return list(ctx.cylinders)
        if cid == "first":
            return ctx.cylinders[:1]
        for sc in ctx.cylinders:
            if sc.id == int(cid):
                return [sc]
        node.get_logger().warn(f"[APPROACH] cylinder_id={cid} not in context - skipping")
        return []

    def tick(self, args, ctx, node):
        s = self._s

        if "targets" not in s:
            targets = self._resolve_targets(args, ctx, node)
            if not targets:
                node.get_logger().warn("[APPROACH] No targets - skipping skill")
                return True
            s["targets"]     = targets
            s["target_idx"]  = -1
            s["phase"]       = "I"
            s["retry_count"] = 0
            node.get_logger().info(
                f"[APPROACH] {len(targets)} target(s) - "
                f"initial settle {self.INITIAL_SETTLE_S:.0f}s")

        standoff = float(args.get("standoff_distance", MIN_STANDOFF_M))

        if s["phase"] == "I":
            done = _settle(node, s, "initial", self.INITIAL_SETTLE_S)
            if done:
                s["target_idx"] = 0
                s["phase"]      = "A"
            return False

        idx = s["target_idx"]
        if idx >= len(s["targets"]):
            node.get_logger().info("[APPROACH] All targets approached")
            return True

        sc = s["targets"][idx]

        def sight_ok() -> bool:
            if node.live_cylinder_center is None:
                return False
            return (time.monotonic() - node._live_center_stamp) < self.LOST_SIGHT_TIMEOUT_S

        def enter_recovery(reason: str):
            s["retry_count"] += 1
            if s["retry_count"] > self.MAX_RETRIES:
                node.get_logger().warn(f"[APPROACH] Cyl {sc.id}: max retries - skipping")
                s["target_idx"] += 1
                s["phase"]       = "A"
                s["retry_count"] = 0
            else:
                node.get_logger().warn(
                    f"[APPROACH] Cyl {sc.id}: {reason} "
                    f"(retry {s['retry_count']}/{self.MAX_RETRIES}) - recovering")
                s["phase"] = "R"

        if s["phase"] == "A":
            ax = sc.world_x - standoff * math.cos(sc.yaw)
            ay = sc.world_y - standoff * math.sin(sc.yaw)
            node.publish_trajectory_setpoint(x=ax, y=ay, z=sc.ned_z, yaw=sc.yaw)
            pos_ok = node.at_position(ax, ay, sc.ned_z)
            yaw_ok = node.at_yaw(sc.yaw)
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Cyl {sc.id} Phase A pos_ok={pos_ok} yaw_ok={yaw_ok}")
            if pos_ok and yaw_ok:
                s["phase"] = "S"
                s.pop("interphase", None)
                node.live_cylinder_center = None
            return False

        if s["phase"] == "S":
            ss = s.setdefault("interphase", {})
            if "end_time" not in ss:
                ss["end_time"] = time.monotonic() + self.APPROACH_SETTLE_S
                cx, cy, cz     = node.current_pos()
                ss["pos"]      = (cx, cy, cz)
                ss["yaw"]      = sc.yaw
            hx, hy, hz = ss["pos"]
            node.publish_trajectory_setpoint(x=hx, y=hy, z=hz, yaw=ss["yaw"])
            remaining = ss["end_time"] - time.monotonic()
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Cyl {sc.id} Phase S settling {remaining:.1f}s")
            if time.monotonic() >= ss["end_time"]:
                s["phase"]               = "B"
                s["center_yaw"]          = sc.yaw
                s["centered_ticks"]      = 0
                s["b_adjust_settle_end"] = 0.0
            return False

        if s["phase"] == "B":
            cx, cy, cz = node.current_pos()
            if not sight_ok():
                enter_recovery("sight lost during centering")
                return False
            now = time.monotonic()
            if now < s.get("b_adjust_settle_end", 0.0):
                node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["center_yaw"])
                return False
            live        = node.live_cylinder_center
            pixel_error = live.x - node._image_half_w
            if abs(pixel_error) <= self.CENTER_PX_TOLERANCE:
                s["centered_ticks"] += 1
            else:
                s["centered_ticks"] = 0
            if s["centered_ticks"] >= self.CENTER_CONFIRM_TICKS:
                s["approach_yaw"] = s["center_yaw"]
                s["phase"]        = "C"
                node.get_logger().info(
                    f"[APPROACH] Cyl {sc.id} Phase B CENTRED "
                    f"yaw={math.degrees(s['center_yaw']):.1f} deg -> Phase C")
                return False
            yaw_delta = self.CENTER_YAW_GAIN * pixel_error
            yaw_delta = math.copysign(min(abs(yaw_delta), self.CENTER_YAW_STEP_MAX), yaw_delta)
            s["center_yaw"]          += yaw_delta
            s["b_adjust_settle_end"]  = now + self.CENTER_ADJUST_SETTLE_S
            node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["center_yaw"])
            return False

        if s["phase"] == "C":
            if not sight_ok():
                enter_recovery("sight lost during depth matching")
                return False
            live_depth = node.live_cylinder_center.z
            cx, cy, cz = node.current_pos()
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Cyl {sc.id} Phase C "
                    f"depth={live_depth:.2f}m target={standoff:.1f}m")
            if abs(live_depth - standoff) < 0.2:
                node.get_logger().info(
                    f"[APPROACH] Cyl {sc.id} Phase C DEPTH MATCHED "
                    f"depth={live_depth:.2f}m standoff={standoff:.1f}m")
                s["target_idx"] += 1
                s["phase"]       = "A"
                s["retry_count"] = 0
                node.live_cylinder_center = None
                if s["target_idx"] >= len(s["targets"]):
                    return True
                return False
            direction = 1.0 if live_depth > standoff else -1.0
            step      = direction * self.DEPTH_STEP_M
            sx = cx + step * math.cos(s["approach_yaw"])
            sy = cy + step * math.sin(s["approach_yaw"])
            node.publish_trajectory_setpoint(x=sx, y=sy, z=sc.ned_z, yaw=s["approach_yaw"])
            return False

        if s["phase"] == "R":
            node.publish_trajectory_setpoint(x=sc.ned_x, y=sc.ned_y, z=sc.ned_z, yaw=sc.yaw)
            if node.at_position(sc.ned_x, sc.ned_y, sc.ned_z) and node.at_yaw(sc.yaw):
                s["phase"] = "S"
                s.pop("interphase", None)
                s["b_adjust_settle_end"] = 0.0
                node.live_cylinder_center = None
            return False

        return False


class MapSkill(Skill):
    """
    Dispatches to either OrbitMapper or VerticalMapper based on args["mode"].
    Each mapper has its own reset/tick interface matching the Skill protocol.
    """

    def __init__(self):
        self._orbit_mapper    = OrbitMapper()
        self._vertical_mapper = VerticalMapper()
        self._active: OrbitMapper | VerticalMapper | None = None

    def reset(self):
        self._orbit_mapper.reset()
        self._vertical_mapper.reset()
        self._active = None

    def tick(self, args, ctx, node):
        if self._active is None:
            mode = args.get("mode", "orbit")
            if mode == "vertical_map":
                self._active = self._vertical_mapper
                node.get_logger().info("[MAP] Mode selected: vertical_map")
            else:
                self._active = self._orbit_mapper
                node.get_logger().info("[MAP] Mode selected: orbit")

        return self._active.tick(args, ctx, node)


# ---------------------------------------------------------------------------
# ORBIT MAPPER  (extracted from original MapSkill)
# ---------------------------------------------------------------------------

class OrbitMapper:
    """Orbits one or all cylinders once per invocation."""

    ORBIT_YAW_STEP: float = 0.015

    def __init__(self):
        self._s: dict     = {}
        self._orbit: dict = {}

    def reset(self):
        self._s     = {}
        self._orbit = {}

    def _resolve_targets(self, args, ctx, node):
        cid = args.get("cylinder_id")
        if cid is None or cid == "all":
            return list(ctx.cylinders)
        for sc in ctx.cylinders:
            if sc.id == int(cid):
                return [sc]
        node.get_logger().warn(f"[MAP/orbit] cylinder_id={cid} not found - skipping")
        return []

    def tick(self, args, ctx, node):
        s = self._s

        if "targets" not in s:
            targets = self._resolve_targets(args, ctx, node)
            if not targets:
                node.get_logger().warn("[MAP/orbit] No targets - skipping skill")
                return True
            nominal_standoff = max(float(args.get("standoff_distance", MIN_STANDOFF_M)), MIN_STANDOFF_M)
            s["targets"]    = targets
            s["target_idx"] = 0
            s["standoff"]   = nominal_standoff
            self._orbit     = {}
            node.get_logger().info(
                f"[MAP/orbit] Orbiting {len(targets)} cylinder(s)  "
                f"(nominal standoff={nominal_standoff:.1f}m)")

        idx = s["target_idx"]
        if idx >= len(s["targets"]):
            node.get_logger().info("[MAP/orbit] All targets mapped")
            return True

        sc       = s["targets"][idx]
        standoff = s["standoff"]

        if not self._orbit or self._orbit.get("for_idx") != idx:
            cx, cy, _ = node.current_pos()
            dx            = cx - sc.world_x
            dy            = cy - sc.world_y
            actual_radius = math.hypot(dx, dy)
            orbit_radius  = max(actual_radius, MIN_STANDOFF_M, standoff)
            start_angle   = math.atan2(dy, dx)

            self._orbit = {
                "for_idx":     idx,
                "angle":       start_angle,
                "accumulated": 0.0,
                "z":           sc.ned_z,
                "radius":      orbit_radius,
                "cx":          sc.world_x,
                "cy":          sc.world_y,
            }
            node.get_logger().info(
                f"[MAP/orbit] Orbit initialised for cyl {sc.id}  "
                f"actual_dist={actual_radius:.2f}m  "
                f"orbit_radius={orbit_radius:.2f}m  "
                f"start_angle={math.degrees(start_angle):.1f} deg")

        o = self._orbit
        o["angle"]       += self.ORBIT_YAW_STEP
        o["accumulated"] += self.ORBIT_YAW_STEP

        theta = o["angle"]
        r     = o["radius"]
        gx    = o["cx"] + r * math.cos(theta)
        gy    = o["cy"] + r * math.sin(theta)
        face_yaw = math.atan2(o["cy"] - gy, o["cx"] - gx)

        node.publish_trajectory_setpoint(x=gx, y=gy, z=o["z"], yaw=face_yaw)

        if node.tick_count % 10 == 0:
            pct = 100.0 * o["accumulated"] / (2 * math.pi)
            node.get_logger().info(
                f"[MAP/orbit] Cyl {sc.id}  {pct:.0f}%  "
                f"angle={math.degrees(theta):.1f} deg  "
                f"r={r:.2f}m  goal=({gx:.2f},{gy:.2f})  "
                f"face_yaw={math.degrees(face_yaw):.1f} deg")

        if o["accumulated"] >= 2 * math.pi:
            node.get_logger().info(
                f"[MAP/orbit] Full orbit complete for cyl {sc.id}  r={r:.2f}m")
            s["target_idx"] += 1
            self._orbit      = {}
            if s["target_idx"] >= len(s["targets"]):
                return True
        return False


# ---------------------------------------------------------------------------
# VERTICAL MAPPER
# ---------------------------------------------------------------------------

class VerticalMapper:
    """
    Performs a systematic vertical sweep around a cylinder in four 90-degree
    arc segments. One full invocation = one complete lap (4 segments).
    Use repeat: N at the plan step level for N laps.

    The cylinder remains in view at ALL times.

    Per-segment phase pipeline:
      SETTLE    - Brief hold at current orbit position so perception stabilises.
      CENTRE    - Yaw-adjust until the cylinder is horizontally centred in frame.
      DESCEND   - Sink slowly to min_altitude_m (AGL) with continuous yaw correction.
      ASCEND    - Rise slowly while monitoring the cylinder's top-edge pixel.
                  Ascent stops when the top edge reaches the image horizontal centre,
                  meaning the camera is now level with the cylinder top.
                  The cylinder stays fully in view throughout; no staleness timers,
                  no hard ceilings.
      ORBIT_QTR - Translate 90 degrees along the orbit circle while facing inward.

    After 4 segments the skill returns True (one full lap complete).

    Cylinder-top detection
    ----------------------
    /geometry/cylinder_info publishes [width_px, height_px, depth_m, ...].
    The estimated top-edge pixel row is:
        top_edge_px = live_cylinder_center.y - live_cylinder_height_px / 2

    When top_edge_px >= _image_half_h the top edge has reached the image centre
    row, indicating the camera eye-level is at or above the cylinder top.
    We use a short confirmation window (TOP_REACHED_CONFIRM_TICKS) so a single
    noisy frame cannot trigger a false stop.
    """

    # Tuning constants -------------------------------------------------------
    DESCEND_SPEED_M_S:      float = 0.15   # m per tick at 10 Hz -> 1.5 cm/tick
    ASCEND_SPEED_M_S:       float = 0.20   # m per tick
    ORBIT_QUARTER_STEP:     float = 0.012  # rad per tick for quarter-orbit translate

    CENTER_PX_TOLERANCE:    float = 25.0   # pixels
    CENTER_CONFIRM_TICKS:   int   = 12
    CENTER_YAW_GAIN:        float = 0.00005
    CENTER_YAW_STEP_MAX:    float = 0.005
    CENTER_SETTLE_S:        float = 0.5

    # How many consecutive ticks the top-edge must be at/above the image center
    # before we consider the cylinder top reached during ascent.
    TOP_REACHED_CONFIRM_TICKS: int = 8

    PHASE_SETTLE_S:         float = 0.8    # settle before CENTRE each segment
    # ------------------------------------------------------------------------

    def __init__(self):
        self._s: dict = {}

    def reset(self):
        self._s = {}

    def _resolve_target(self, args, ctx, node) -> SpottedCylinder | None:
        cid = args.get("cylinder_id")
        if cid is None or cid == "all":
            if ctx.cylinders:
                return ctx.cylinders[0]
            return None
        for sc in ctx.cylinders:
            if sc.id == int(cid):
                return sc
        node.get_logger().warn(f"[MAP/vmap] cylinder_id={cid} not found - skipping")
        return None

    def tick(self, args, ctx, node) -> bool:
        s = self._s

        # ------------------------------------------------------------------ #
        #  INITIALIZATION                                                     #
        # ------------------------------------------------------------------ #
        if "sc" not in s:
            sc = self._resolve_target(args, ctx, node)
            if sc is None:
                node.get_logger().warn("[MAP/vmap] No target - skipping skill")
                return True

            standoff = float(args.get("standoff_distance", 5.0))
            standoff = max(VMAP_MIN_STANDOFF_M, min(standoff, VMAP_MAX_STANDOFF_M))
            min_alt  = float(args.get("min_altitude_m", VMAP_MIN_ALTITUDE_M))
            min_alt  = max(min_alt, VMAP_MIN_ALTITUDE_M)

            # Determine starting angle from current drone position
            cx, cy, _ = node.current_pos()
            dx = cx - sc.world_x
            dy = cy - sc.world_y
            actual_radius = math.hypot(dx, dy)
            orbit_radius  = max(actual_radius, standoff, VMAP_MIN_STANDOFF_M)
            start_angle   = math.atan2(dy, dx)

            s.update({
                "sc":            sc,
                "standoff":      standoff,
                "orbit_radius":  orbit_radius,
                "min_alt":       min_alt,
                "segment":       0,          # 0-3: which quarter we are on
                "phase":         "SETTLE",   # SETTLE -> CENTRE -> DESCEND -> ASCEND -> ORBIT_QTR
                "angle":         start_angle,
                "center_yaw":    math.atan2(sc.world_y - cy, sc.world_x - cx),
                "centered_ticks": 0,
                "center_settle_end": 0.0,
            })
            node.get_logger().info(
                f"[MAP/vmap] Init cyl={sc.id}  standoff={standoff:.1f}m  "
                f"min_alt={min_alt:.1f}m  orbit_r={orbit_radius:.2f}m  "
                f"start_angle={math.degrees(start_angle):.1f} deg")

        sc            = s["sc"]
        standoff      = s["standoff"]
        orbit_radius  = s["orbit_radius"]
        min_alt       = s["min_alt"]
        cx, cy, cz    = node.current_pos()

        # Helper: position on orbit circle at current angle
        def orbit_xy(angle: float):
            gx = sc.world_x + orbit_radius * math.cos(angle)
            gy = sc.world_y + orbit_radius * math.sin(angle)
            return gx, gy

        # Helper: yaw to face cylinder centre from a given (x, y)
        def face_yaw(gx: float, gy: float) -> float:
            return math.atan2(sc.world_y - gy, sc.world_x - gx)

        # ------------------------------------------------------------------ #
        #  SETTLE  (brief hold at start of each segment before centering)    #
        # ------------------------------------------------------------------ #
        if s["phase"] == "SETTLE":
            if "settle_end" not in s:
                s["settle_end"] = time.monotonic() + self.PHASE_SETTLE_S
                gx, gy = orbit_xy(s["angle"])
                s["hold_pos"] = (gx, gy, cz)
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} SETTLE {self.PHASE_SETTLE_S:.1f}s  "
                    f"angle={math.degrees(s['angle']):.1f} deg")

            hx, hy, hz = s["hold_pos"]
            fy = face_yaw(hx, hy)
            node.publish_trajectory_setpoint(x=hx, y=hy, z=hz, yaw=fy)

            if time.monotonic() >= s["settle_end"]:
                s.pop("settle_end", None)
                s["phase"]           = "CENTRE"
                s["centered_ticks"]  = 0
                s["center_yaw"]      = face_yaw(hx, hy)
                s["center_settle_end"] = 0.0
                node.live_cylinder_center = None
            return False

        # ------------------------------------------------------------------ #
        #  CENTRE  (fine yaw alignment so cylinder is in frame centre)       #
        # ------------------------------------------------------------------ #
        if s["phase"] == "CENTRE":
            gx, gy = orbit_xy(s["angle"])
            fy     = s["center_yaw"]

            # If no live detection yet, hold position and wait
            if node.live_cylinder_center is None:
                node.publish_trajectory_setpoint(x=gx, y=gy, z=cz, yaw=fy)
                return False

            now = time.monotonic()
            if now < s.get("center_settle_end", 0.0):
                node.publish_trajectory_setpoint(x=gx, y=gy, z=cz, yaw=fy)
                return False

            pixel_error = node.live_cylinder_center.x - node._image_half_w
            if abs(pixel_error) <= self.CENTER_PX_TOLERANCE:
                s["centered_ticks"] += 1
            else:
                s["centered_ticks"] = 0

            if s["centered_ticks"] >= self.CENTER_CONFIRM_TICKS:
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} CENTRED  "
                    f"yaw={math.degrees(fy):.1f} deg -> DESCEND")
                s["phase"]             = "DESCEND"
                s["descend_target_z"]  = -min_alt   # NED z is negative-up
                s["hold_xy"]           = (gx, gy)
                s["hold_yaw"]          = fy
                return False

            yaw_delta = self.CENTER_YAW_GAIN * pixel_error
            yaw_delta = math.copysign(min(abs(yaw_delta), self.CENTER_YAW_STEP_MAX), yaw_delta)
            s["center_yaw"]        += yaw_delta
            s["center_settle_end"]  = now + self.CENTER_SETTLE_S
            node.publish_trajectory_setpoint(x=gx, y=gy, z=cz, yaw=s["center_yaw"])
            return False

        # ------------------------------------------------------------------ #
        #  DESCEND  (sink slowly to min_alt while holding horizontal pos)    #
        # ------------------------------------------------------------------ #
        if s["phase"] == "DESCEND":
            hx, hy   = s["hold_xy"]
            target_z = s["descend_target_z"]   # NED (negative = lower altitude)

            # Step downward in NED (z increases toward zero = descend)
            new_z = cz + self.DESCEND_SPEED_M_S
            new_z = min(new_z, target_z)   # don't overshoot the floor

            # Continuous yaw correction to keep cylinder centred
            if node.live_cylinder_center is not None:
                pixel_error   = node.live_cylinder_center.x - node._image_half_w
                yaw_delta     = self.CENTER_YAW_GAIN * pixel_error
                yaw_delta     = math.copysign(min(abs(yaw_delta), self.CENTER_YAW_STEP_MAX), yaw_delta)
                s["hold_yaw"] += yaw_delta

            node.publish_trajectory_setpoint(x=hx, y=hy, z=new_z, yaw=s["hold_yaw"])

            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} DESCEND  "
                    f"alt={-new_z:.2f}m  target={min_alt:.2f}m")

            if abs(new_z - target_z) < 0.1:
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} DESCEND complete  alt={min_alt:.2f}m -> ASCEND")
                s["phase"]             = "ASCEND"
                s["top_reached_ticks"] = 0
            return False

        # ------------------------------------------------------------------ #
        #  ASCEND  (rise until camera is level with cylinder top)            #
        #                                                                     #
        #  Termination criterion (cylinder stays in view the entire time):   #
        #    top_edge_px = live_cylinder_center.y - live_cylinder_height_px/2#
        #    Stop when top_edge_px >= _image_half_h for                      #
        #    TOP_REACHED_CONFIRM_TICKS consecutive ticks.                     #
        #                                                                     #
        #  When the top edge reaches the image horizontal centre the camera  #
        #  eye-level is at the cylinder top. No staleness timers; no hard    #
        #  ceilings; the cylinder is always visible.                         #
        # ------------------------------------------------------------------ #
        if s["phase"] == "ASCEND":
            hx, hy = s["hold_xy"]

            # Step upward (NED z decreases = altitude increases)
            new_z = cz - self.ASCEND_SPEED_M_S

            # Continuous yaw correction to keep cylinder centred horizontally
            if node.live_cylinder_center is not None:
                pixel_error   = node.live_cylinder_center.x - node._image_half_w
                yaw_delta     = self.CENTER_YAW_GAIN * pixel_error
                yaw_delta     = math.copysign(min(abs(yaw_delta), self.CENTER_YAW_STEP_MAX), yaw_delta)
                s["hold_yaw"] += yaw_delta

            node.publish_trajectory_setpoint(x=hx, y=hy, z=new_z, yaw=s["hold_yaw"])

            # --- Cylinder-top detection ---
            # Evaluate top-edge position only when we have a fresh detection.
            top_reached = False
            if node.live_cylinder_center is not None and node.live_cylinder_height_px > 0:
                cy_px      = node.live_cylinder_center.y          # pixel row of centroid
                h_px       = node.live_cylinder_height_px         # apparent height in pixels
                top_edge   = cy_px - h_px / 2.0                   # estimated top-edge row
                # top_edge counts DOWN from image top (row 0 = top of frame).
                # _image_half_h is the row of the image horizontal centre.
                # When top_edge >= _image_half_h the top of the cylinder has
                # descended to (or below) centre, meaning the camera is at or
                # above the cylinder top.
                if top_edge >= node._image_half_h:
                    s["top_reached_ticks"] += 1
                else:
                    s["top_reached_ticks"] = 0  # reset if a single frame disagrees

                top_reached = s["top_reached_ticks"] >= self.TOP_REACHED_CONFIRM_TICKS

                if node.tick_count % 10 == 0:
                    node.get_logger().info(
                        f"[MAP/vmap] Seg {s['segment']} ASCEND  "
                        f"alt={-new_z:.2f}m  "
                        f"top_edge={top_edge:.0f}px  "
                        f"img_centre={node._image_half_h:.0f}px  "
                        f"confirm={s['top_reached_ticks']}/{self.TOP_REACHED_CONFIRM_TICKS}")
            else:
                if node.tick_count % 10 == 0:
                    node.get_logger().info(
                        f"[MAP/vmap] Seg {s['segment']} ASCEND  "
                        f"alt={-new_z:.2f}m  (awaiting detection)")

            if top_reached:
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} ASCEND complete - "
                    f"camera level with cylinder top at alt={-new_z:.2f}m -> ORBIT_QTR")
                s["phase"]           = "ORBIT_QTR"
                s["qtr_accumulated"] = 0.0
                s["top_reached_ticks"] = 0
            return False

        # ------------------------------------------------------------------ #
        #  ORBIT_QTR  (translate 90 deg along orbit circle, face inward)    #
        # ------------------------------------------------------------------ #
        if s["phase"] == "ORBIT_QTR":
            quarter = math.pi / 2.0

            s["angle"]          += self.ORBIT_QUARTER_STEP
            s["qtr_accumulated"] += self.ORBIT_QUARTER_STEP

            gx, gy   = orbit_xy(s["angle"])
            fy        = face_yaw(gx, gy)
            node.publish_trajectory_setpoint(x=gx, y=gy, z=cz, yaw=fy)

            if node.tick_count % 10 == 0:
                pct = 100.0 * s["qtr_accumulated"] / quarter
                node.get_logger().info(
                    f"[MAP/vmap] Seg {s['segment']} ORBIT_QTR  {pct:.0f}%  "
                    f"angle={math.degrees(s['angle']):.1f} deg")

            if s["qtr_accumulated"] >= quarter:
                s["segment"] += 1
                node.get_logger().info(
                    f"[MAP/vmap] Quarter-orbit complete -> segment {s['segment']}")

                if s["segment"] >= 4:
                    # Full lap done
                    node.get_logger().info("[MAP/vmap] Full vertical-map lap COMPLETE")
                    return True

                # Start next segment
                s["phase"]          = "SETTLE"
                s["centered_ticks"] = 0
                s["center_yaw"]     = face_yaw(*orbit_xy(s["angle"]))
                s["qtr_accumulated"] = 0.0
                node.live_cylinder_center = None
            return False

        # Fallback
        return False


class ReturnHomeSkill(Skill):
    def __init__(self):
        self._sent = False

    def reset(self):
        self._sent = False

    def tick(self, args, ctx, node):
        if not self._sent:
            node.send_return_to_launch()
            self._sent = True
            node.get_logger().info(
                "[RETURN_HOME] RTL sent\n"
                "=== MISSION CYLINDER SUMMARY ===\n"
                + _spotted_summary(ctx.cylinders)
            )
        return True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _spotted_summary(cylinders: list) -> str:
    if not cylinders:
        return "  No cylinders spotted."
    lines = [f"  {len(cylinders)} cylinder(s):"]
    for sc in cylinders:
        lines.append(
            f"    [ID {sc.id:02d}]  "
            f"drone_pos=({sc.ned_x:+.2f},{sc.ned_y:+.2f},{sc.ned_z:+.2f})  "
            f"yaw={math.degrees(sc.yaw):+.1f} deg  depth={sc.depth_m:.2f}m  "
            f"world=({sc.world_x:+.2f},{sc.world_y:+.2f})"
        )
    return "\n".join(lines)


def _settle(node, state_dict: dict, key: str, duration_s: float) -> bool:
    ss = state_dict.setdefault(key, {})
    if "end_time" not in ss:
        cx, cy, cz = node.current_pos()
        yaw         = node.current_yaw()
        ss["end_time"] = time.monotonic() + duration_s
        ss["pos"]      = (cx, cy, cz)
        ss["yaw"]      = yaw
        node.get_logger().info(
            f"[SETTLE/{key}] Holding {duration_s:.1f}s at "
            f"({cx:.2f},{cy:.2f},{cz:.2f}) yaw={math.degrees(yaw):.1f} deg")
    hx, hy, hz = ss["pos"]
    node.publish_trajectory_setpoint(x=hx, y=hy, z=hz, yaw=ss["yaw"])
    remaining = ss["end_time"] - time.monotonic()
    if node.tick_count % 10 == 0:
        node.get_logger().info(f"[SETTLE/{key}] {remaining:.1f}s remaining")
    return time.monotonic() >= ss["end_time"]


# ---------------------------------------------------------------------------
# SKILL REGISTRY
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, type[Skill]] = {
    "takeoff":      TakeoffSkill,
    "search":       SearchYawScanSkill,
    "approach":     ApproachSkill,
    "map":          MapSkill,
    "return_home":  ReturnHomeSkill,
}


# ---------------------------------------------------------------------------
# MID-MISSION AUTOMATIC REPLANNER
# ---------------------------------------------------------------------------

def _extract_mapped_cylinders(completed_steps: list[dict]) -> dict[int, int]:
    mapped: dict[int, int] = {}
    for step in completed_steps:
        if not isinstance(step, dict):
            continue
        if step.get("state") == "map":
            cid = step.get("args", {}).get("cylinder_id")
            if cid is not None and cid != "all":
                cid    = int(cid)
                passes = step.get("repeat", 1)
                mapped[cid] = mapped.get(cid, 0) + passes
    return mapped


def _parse_replan_response(raw: str) -> tuple[list | None, str | None]:
    stripped = raw.strip()

    has_nominal   = bool(re.search(r'\bNOMINAL\b', stripped, re.IGNORECASE))
    has_json_plan = bool(re.search(r'\[\s*\{', stripped))

    if has_nominal and not has_json_plan:
        return None, None

    if has_nominal and has_json_plan:
        import logging
        logging.getLogger(__name__).warning(
            "_parse_replan_response: response contains both NOMINAL and a JSON "
            "array - treating as a revised plan (ignoring NOMINAL token).")

    reason = None
    reason_match = re.search(r"REASON:\s*(.+)$", stripped, re.IGNORECASE | re.MULTILINE)
    if reason_match:
        reason = reason_match.group(1).strip()

    spans: list[tuple[int, int]] = []
    depth = 0
    span_start = -1
    for i, ch in enumerate(stripped):
        if ch == "[":
            if depth == 0:
                span_start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and span_start != -1:
                spans.append((span_start, i))
                span_start = -1

    if not spans:
        raise ValueError("No JSON array found in replanner response")

    best_start, best_end = max(spans, key=lambda s: s[1] - s[0])
    json_portion = stripped[best_start : best_end + 1]
    tail = extract_json(json_portion)
    return tail, reason


class AutoReplanner:
    def __init__(self):
        self._running = False
        self._lock    = threading.Lock()

    def trigger(self, node, completed_step_name: str) -> None:
        if completed_step_name in _NO_REPLAN_AFTER:
            node.get_logger().info(
                f"[REPLAN] Skipping replan after '{completed_step_name}' "
                f"(no new discoveries possible)")
            node._replan_pending = False
            return

        with self._lock:
            if self._running:
                node.get_logger().info(
                    "[REPLAN] Previous replan still running - skipping trigger")
                node._replan_pending = False
                return
            self._running = True

        mission_intent = node.mission_intent

        def _worker():
            try:
                current_step    = node.current_step
                completed_steps = list(node.plan[:current_step])
                remaining_steps = list(node.plan[current_step:])
                cylinders       = list(node.ctx.cylinders)

                system_prompt = build_replan_system_prompt(
                    mission_intent  = mission_intent,
                    completed_steps = completed_steps,
                    remaining_steps = remaining_steps,
                    cylinders       = cylinders,
                )

                node.get_logger().info(
                    "\n" + "="*80 +
                    "\n[REPLAN PROMPT SENT TO LLM]\n" +
                    system_prompt +
                    "\n" + "="*80
                )

                node.get_logger().info(
                    f"[REPLAN] Querying LLM ({LLM_BACKEND}/{GEMINI_MODEL if LLM_BACKEND == 'gemini' else CEREBRAS_MODEL})  "
                    f"({len(cylinders)} cylinder(s) known, "
                    f"{len(remaining_steps)} step(s) remaining)")

                tail   = None
                reason = None
                for attempt in range(3):
                    try:
                        raw = query_llm("", system=system_prompt, max_tokens=4096)
                        node.get_logger().info(
                            f"[REPLAN] Raw response (attempt {attempt+1}):\n{raw}")

                        tail, reason = _parse_replan_response(raw)

                        if tail is None:
                            node.get_logger().info(
                                "[REPLAN] LLM returned NOMINAL - plan unchanged")
                            return

                        ok, errs = validate_plan(tail, is_tail=True)
                        if ok:
                            break

                        err_str = "; ".join(errs)
                        node.get_logger().warn(
                            f"[REPLAN] Invalid tail (attempt {attempt+1}): {err_str}")

                        system_prompt = (
                            system_prompt
                            + f"\n\n--- PREVIOUS ATTEMPT {attempt+1} WAS INVALID ---\n"
                            + "Errors found:\n"
                            + "\n".join(f"  - {e}" for e in errs)
                            + "\n\nFix all errors and output the corrected JSON array "
                              "followed by REASON: <one sentence>."
                        )
                        tail   = None
                        reason = None

                    except Exception as exc:
                        node.get_logger().warn(
                            f"[REPLAN] Attempt {attempt+1} exception: {exc}")

                if tail is None:
                    node.get_logger().warn(
                        "[REPLAN] Could not produce a valid tail after 3 attempts - "
                        "leaving plan unchanged")
                    return

                live_step = node.current_step
                head      = list(node.plan[:live_step])
                node.plan = head + tail

                reason_str = f"  Reason: {reason}" if reason else ""
                print_plan(
                    node.plan,
                    logger       = node.get_logger(),
                    current_step = live_step,
                    label        = (
                        f"REVISED PLAN  ({len(tail)}-step tail from replanner)"
                        + (f"\n  {reason_str}" if reason_str else "")
                    ),
                )

            finally:
                node._replan_pending = False
                with self._lock:
                    self._running = False

        threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# EXECUTOR NODE
# ---------------------------------------------------------------------------

class MissionExecutorNode(Node):

    POSITION_THRESHOLD   = 0.3
    YAW_THRESHOLD        = 0.05
    SPOTTED_MERGE_RADIUS = 1.2
    TRANSITION_SETTLE_S  = 4.0

    def __init__(self, plan: list, mission_intent: str):
        super().__init__("mission_executor_node")

        self.plan           = plan
        self.mission_intent = mission_intent
        self.current_step   = 0
        self.tick_count     = 0

        self._skill_instances: dict[str, Skill] = {
            name: cls() for name, cls in SKILL_REGISTRY.items()
        }
        self._active_skill:   Skill | None = None
        self._repeat_count:   int          = 0
        self._skill_repeats:  int          = 1
        self._step_done:      bool         = False
        self._step_done_name: str          = ""
        self._settle_state:   dict         = {}

        self._replanner             = AutoReplanner()
        self._replan_pending: bool  = False

        self.ctx = ExecutionContext()

        self.odometry       = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.spotted: list[SpottedCylinder] = []
        self._spotted_id_counter            = 1
        self.live_cylinder_center: Point | None = None
        self._live_center_stamp: float          = 0.0
        self._image_half_w: float               = 400.0
        self._image_half_h: float               = 300.0   # updated via /geometry/image_size
        # Live cylinder apparent height in pixels from /geometry/cylinder_info.
        # Used by VerticalMapper to detect when the drone is level with the
        # cylinder top (top-edge pixel = cy_px - height_px/2 reaches image centre).
        self.live_cylinder_height_px: float     = 0.0

        self.offboard_setpoint_counter = 0
        self.armed_and_offboard        = False

        qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability  = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 1,
        )

        self.vehicle_command_pub       = self.create_publisher(
            VehicleCommand,      "/fmu/in/vehicle_command",       qos)
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.trajectory_setpoint_pub   = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",   qos)
        self.search_active_pub         = self.create_publisher(
            Bool, "/mission/search_active", 10)

        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry",
            self._odometry_cb, qos)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, qos)
        self.create_subscription(
            Float32MultiArray, "/geometry/confirmed_cylinder",
            self._confirmed_cylinder_cb, 10)
        self.create_subscription(
            Point, "/geometry/cylinder_center",
            self._live_center_cb, 10)
        self.create_subscription(
            Int32MultiArray, "/geometry/image_size",
            self._image_size_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/geometry/cylinder_info",
            self._cylinder_info_cb, 10)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f"Mission executor initialised  "
            f"[LLM backend: {LLM_BACKEND} / "
            f"{GEMINI_MODEL if LLM_BACKEND == 'gemini' else CEREBRAS_MODEL}]")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odometry_cb(self, msg):
        self.odometry = msg

    def _status_cb(self, msg):
        self.vehicle_status = msg

    def _live_center_cb(self, msg):
        self.live_cylinder_center = msg
        self._live_center_stamp   = time.monotonic()

    def _image_size_cb(self, msg):
        if len(msg.data) >= 1:
            self._image_half_w = float(msg.data[0]) / 2.0
        if len(msg.data) >= 2:
            self._image_half_h = float(msg.data[1]) / 2.0

    def _cylinder_info_cb(self, msg: Float32MultiArray):
        # /geometry/cylinder_info layout: [width_px, height_px, depth_m, 1.0, 1.0]
        if len(msg.data) >= 2:
            self.live_cylinder_height_px = float(msg.data[1])

    def _confirmed_cylinder_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            return
        cx_px, cy_px, depth_m, width_px = (float(v) for v in msg.data[:4])
        if depth_m <= 0.0 or math.isnan(depth_m):
            return
        ned_x, ned_y, ned_z = self.current_pos()
        yaw = self.current_yaw()
        pixel_error     = cx_px - self._image_half_w
        CENTER_YAW_GAIN = 0.00005
        yaw_corrected   = yaw + CENTER_YAW_GAIN * pixel_error * 5.0
        world_x = ned_x + depth_m * math.cos(yaw_corrected)
        world_y = ned_y + depth_m * math.sin(yaw_corrected)

        for sc in self.spotted:
            dist = math.hypot(world_x - sc.world_x, world_y - sc.world_y)
            if dist < self.SPOTTED_MERGE_RADIUS:
                if depth_m < sc.depth_m:
                    sc.depth_m = depth_m
                    sc.ned_x = ned_x; sc.ned_y = ned_y; sc.ned_z = ned_z
                    sc.yaw   = yaw_corrected
                    sc.world_x = world_x; sc.world_y = world_y
                return

        sc = SpottedCylinder(
            id=self._spotted_id_counter,
            ned_x=ned_x, ned_y=ned_y, ned_z=ned_z,
            yaw=yaw_corrected, depth_m=depth_m,
            world_x=world_x, world_y=world_y,
            px_cx=cx_px, px_cy=cy_px, width_px=width_px,
        )
        self.spotted.append(sc)
        self._spotted_id_counter += 1
        self.get_logger().info(
            f"[SPOTTED] Cyl ID={sc.id}  "
            f"depth={depth_m:.2f}m  world=({world_x:.2f},{world_y:.2f})")

    # ------------------------------------------------------------------
    # PX4 helpers
    # ------------------------------------------------------------------

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        for k in range(1, 8):
            setattr(msg, f"param{k}", float(params.get(f"param{k}", 0.0)))
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = msg.acceleration = msg.attitude = msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        msg          = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw      = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def send_return_to_launch(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    def _publish_search_active(self, active: bool):
        msg = Bool(); msg.data = active
        self.search_active_pub.publish(msg)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def current_pos(self) -> tuple[float, float, float]:
        try:
            return tuple(float(v) for v in self.odometry.position[:3])
        except Exception:
            return (0.0, 0.0, 0.0)

    def current_altitude(self) -> float:
        return -self.current_pos()[2]

    def current_yaw(self) -> float:
        try:
            q    = self.odometry.q
            siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
            cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
            return math.atan2(siny, cosy)
        except Exception:
            return 0.0

    def at_position(self, tx, ty, tz) -> bool:
        x, y, z = self.current_pos()
        return math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2) < self.POSITION_THRESHOLD

    def at_yaw(self, target: float) -> bool:
        diff = abs(self.current_yaw() - target)
        return min(diff, 2 * math.pi - diff) < self.YAW_THRESHOLD

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        self.publish_offboard_control_mode()
        self.tick_count += 1

        if self.offboard_setpoint_counter == 10 and not self.armed_and_offboard:
            self.engage_offboard_mode()
            self.arm()
            self.armed_and_offboard = True
        self.offboard_setpoint_counter += 1

        if not self.armed_and_offboard:
            return
        if self.current_step >= len(self.plan):
            return

        step = self.plan[self.current_step]

        if not isinstance(step, dict):
            self.get_logger().error(
                f"[CONTROL] Step {self.current_step} is not a dict "
                f"({type(step).__name__}: {step!r}) - skipping")
            self.current_step += 1
            return

        if self._step_done:
            settled = _settle(self, self._settle_state, "_transition",
                              self.TRANSITION_SETTLE_S)
            if settled:
                completed_name = self._step_done_name

                self.current_step  += 1
                self._step_done     = False
                self._settle_state  = {}
                self._active_skill  = None

                self._replan_pending = True
                self._replanner.trigger(self, completed_step_name=completed_name)

                self.get_logger().info(
                    f"Step {self.current_step - 1} [{completed_name}] "
                    f"transition complete - advancing to step {self.current_step}")
                print_plan(
                    self.plan,
                    logger       = self.get_logger(),
                    current_step = self.current_step,
                    label        = f"PLAN PROGRESS  (step {self.current_step}/{len(self.plan)})",
                )
            return

        if self._replan_pending:
            if self.tick_count % 20 == 0:
                self.get_logger().info(
                    f"[CONTROL] Waiting for replanner before activating "
                    f"step {self.current_step} [{self.plan[self.current_step].get('state', '?')}]")
            return

        skill_name = step["state"]
        if self._active_skill is None:
            skill = self._skill_instances.get(skill_name)
            if skill is None:
                self.get_logger().warn(f"Unknown skill '{skill_name}' - skipping")
                self._step_done      = True
                self._step_done_name = skill_name
                return

            repeat = step.get("repeat", 1)
            self._active_skill  = skill
            self._repeat_count  = 0
            self._skill_repeats = repeat
            skill.reset()
            self.get_logger().info(
                f"Step {self.current_step}: starting [{skill_name}]  "
                f"repeats={repeat}")

        done = self._active_skill.tick(step.get("args", {}), self.ctx, self)

        if done:
            self._repeat_count += 1
            self.get_logger().info(
                f"[REPEAT] step={self.current_step} [{skill_name}]  "
                f"rep {self._repeat_count}/{self._skill_repeats} complete")

            if self._repeat_count < self._skill_repeats:
                self._active_skill.reset()
                self.get_logger().info(
                    f"Step {self.current_step} [{skill_name}] "
                    f"rep {self._repeat_count}/{self._skill_repeats} done "
                    f"- resetting for next pass")
            else:
                self.get_logger().info(
                    f"Step {self.current_step} [{skill_name}] "
                    f"all {self._skill_repeats} repetition(s) complete "
                    f"- settling {self.TRANSITION_SETTLE_S:.0f}s")
                self._step_done      = True
                self._step_done_name = skill_name
                self._settle_state   = {}


# ---------------------------------------------------------------------------
# PLANNING + ENTRY POINT
# ---------------------------------------------------------------------------

def generate_plan(mission: str) -> list | None:
    print(f"\n{'='*60}")
    print(f"Mission: {mission}")
    print("="*60)
    for attempt in range(3):
        try:
            raw  = query_llm(mission, system=PLANNING_PROMPT, max_tokens=4096)
            print(f"  [RAW attempt {attempt+1}]:\n{raw}\n")
            plan = extract_json(raw)
            print("  [STEP repeat values]:")
            for i, step in enumerate(plan):
                if not isinstance(step, dict):
                    print(f"    [{i}] <non-dict: {step!r}>")
                    continue
                rep      = step.get("repeat", 1)
                args_rep = step.get("args", {}).get("repeat", None)
                flag     = "  <-- MISPLACED!" if args_rep is not None else ""
                print(f"    [{i}] {step['state']:22} repeat={rep}{flag}")
            print()
            ok, errs = validate_plan(plan)
            if ok:
                print_plan(plan, label=f"INITIAL PLAN  ({len(plan)} steps)")
                return plan
            print(f"  [INVALID]:", "; ".join(errs))
        except Exception as exc:
            print(f"  [ERROR] {exc}")
    return None


def main():
    if not MISSIONS:
        print("No missions defined - aborting.")
        return
    mission = MISSIONS[0]
    plan    = generate_plan(mission)
    if plan is None:
        print("Could not generate a valid plan - aborting.")
        return
    print(f"\nExecuting {len(plan)}-step plan for: {mission!r}")
    rclpy.init()
    node = MissionExecutorNode(plan, mission_intent=mission)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
