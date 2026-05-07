"""
experiment_utils.py
-------------------
Shared utilities for all UAV planning/replanning experiment scripts.

Provides:
  - LLM client wrappers (Gemini, Cerebras) with the same interface
  - Planning prompt and plan generation
  - Replan prompt builder and response parser (mirrors mission_executor.py logic)
  - Plan validator
  - MockWorldState for injecting synthetic failure scenarios without ROS2/Gazebo
  - Result logging helpers

Rate-limit / 429 handling
--------------------------
  ALL backends (Gemini and Cerebras) now retry indefinitely on rate-limit
  and transient 5xx errors.  The retry strategy is:

  Gemini:
    - Multiple API keys are held in GEMINI_API_KEYS (pool).
    - On a 429 / quota / rate-limit error the code immediately rotates to the
      next key in the pool.
    - After exhausting ALL keys in one sweep it waits INTER_KEY_WAIT_S seconds
      then starts a fresh sweep.
    - This outer wait-and-retry loop repeats indefinitely until a real response
      (success or a genuine non-transient error) is obtained.

  Cerebras (and any non-Gemini backend):
    - On a rate-limit / 429 / 503 / transient error the code waits
      CEREBRAS_RETRY_WAIT_S seconds and retries the same call.
    - The retry loop also repeats indefinitely.

  In BOTH cases:
    - Rate-limit retries are TRANSPARENT to callers: they never count as a
      failed planning attempt and never appear in result error lists.
    - Wall-clock time spent waiting IS included in the returned latency.
    - Only genuine model-logic failures (bad JSON, constraint violations) are
      surfaced as errors to callers.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

MODELS = {
    "gemini-flash-2.5": {
        "backend":    "gemini",
        "model_id":   "gemini-2.5-flash",
    },
    "qwen-235b": {
        "backend":    "cerebras",
        "model_id":   "qwen-3-235b-a22b-instruct-2507",
    },
    "llama-3.1-8b": {
        "backend":    "cerebras",
        "model_id":   "llama3.1-8b",
    },
}

# ---------------------------------------------------------------------------
# GEMINI KEY POOL
# ---------------------------------------------------------------------------
# Keys are tried in order; on a rate-limit hit the code immediately moves to
# the next key.  When all keys have been tried once, it waits INTER_KEY_WAIT_S
# before starting the next rotation sweep.

GEMINI_API_KEYS: list[str] = [
    k for k in [
        os.environ.get("GEMINI_API_KEY_1", "X"),
        os.environ.get("GEMINI_API_KEY_2", "X"),
        os.environ.get("GEMINI_API_KEY_3", "X"),
    ] if k  # drop empty strings
]

# How long to wait (seconds) after exhausting all Gemini keys before retrying.
INTER_KEY_WAIT_S: int = 30

# How long to wait (seconds) between Cerebras rate-limit retries.
CEREBRAS_RETRY_WAIT_S: int = 15

# Cap on individual wait between retries to avoid runaway sleeps on 5xx storms.
MAX_SINGLE_WAIT_S: int = 120

CEREBRAS_API_KEY = os.environ.get(
    "CEREBRAS_API_KEY",
    "X",
)

# ---------------------------------------------------------------------------
# CLIENT CACHE
# One cached client object per Gemini API key, plus one Cerebras client.
# ---------------------------------------------------------------------------

_gemini_clients: dict[str, Any] = {}   # key_string -> genai.Client
_cerebras_client = None


def _get_gemini_client(api_key: str):
    """Return (and lazily create) a Gemini client for the given API key."""
    if api_key not in _gemini_clients:
        from google import genai as _genai
        _gemini_clients[api_key] = _genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def _get_cerebras():
    global _cerebras_client
    if _cerebras_client is None:
        from cerebras.cloud.sdk import Cerebras as _Cerebras
        _cerebras_client = _Cerebras(api_key=CEREBRAS_API_KEY)
    return _cerebras_client


# ---------------------------------------------------------------------------
# RATE-LIMIT / TRANSIENT ERROR DETECTION
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Return True for any error that warrants a transparent retry rather than
    surfacing as a genuine planning failure.

    Covers:
      - 429 Too Many Requests  (all backends)
      - 503 Service Unavailable / overloaded
      - Quota exhausted / resource exhausted
      - "High traffic" / "try again" messages
      - Connection-level transient errors
    """
    msg = str(exc).lower()
    return any(
        kw in msg for kw in (
            "429",
            "quota",
            "rate",
            "resource_exhausted",
            "too many requests",
            "503",
            "service unavailable",
            "overloaded",
            "high traffic",
            "try again",
            "temporarily unavailable",
            "server error",
            "internal error",
            "connection",
            "timeout",
            "timed out",
        )
    )


# ---------------------------------------------------------------------------
# LLM QUERY  (transparent retry for ALL backends)
# ---------------------------------------------------------------------------

def query_llm(
    prompt: str,
    system: str,
    model_key: str,
    max_tokens: int = 4096,
) -> tuple[str, float]:
    """
    Call the LLM identified by model_key.
    Returns (response_text, latency_seconds).

    Rate-limit and transient errors are retried indefinitely and
    TRANSPARENTLY — they never surface as exceptions to callers.

    Gemini:
      Rotates through GEMINI_API_KEYS on each 429.  After exhausting the full
      pool waits INTER_KEY_WAIT_S seconds before the next sweep.

    Cerebras (and any non-Gemini backend):
      Waits CEREBRAS_RETRY_WAIT_S seconds between retries, doubling on
      consecutive failures up to MAX_SINGLE_WAIT_S.
    """
    cfg      = MODELS[model_key]
    backend  = cfg["backend"]
    model_id = cfg["model_id"]

    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Cerebras — exponential backoff on rate-limit / transient errors
    # ------------------------------------------------------------------
    if backend != "gemini":
        wait_s = CEREBRAS_RETRY_WAIT_S
        attempt = 0
        while True:
            attempt += 1
            try:
                client = _get_cerebras()
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=0,
                    stream=False,
                )
                return resp.choices[0].message.content, time.monotonic() - t0

            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    # Genuine non-transient error — propagate to caller.
                    raise
                elapsed = round(time.monotonic() - t0, 1)
                print(
                    f"  [{model_key}] Transient/rate-limit error on attempt {attempt} "
                    f"(+{elapsed}s elapsed): {exc!r:.120}. "
                    f"Waiting {wait_s}s before retry..."
                )
                time.sleep(wait_s)
                wait_s = min(wait_s * 2, MAX_SINGLE_WAIT_S)

    # ------------------------------------------------------------------
    # Gemini — pool-based key rotation with inter-sweep wait
    # ------------------------------------------------------------------
    from google.genai import types as _genai_types

    if not GEMINI_API_KEYS:
        raise RuntimeError("No Gemini API keys configured in GEMINI_API_KEYS.")

    key_index  = 0
    sweep_hits = 0   # rate-limit hits in the current sweep

    while True:
        api_key = GEMINI_API_KEYS[key_index % len(GEMINI_API_KEYS)]
        client  = _get_gemini_client(api_key)

        config = _genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
        )

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt if prompt else "Respond now.",
                config=config,
            )
            # SUCCESS
            return response.text, time.monotonic() - t0

        except Exception as exc:
            if not _is_rate_limit_error(exc):
                # Non-transient error: propagate immediately.
                raise

            sweep_hits += 1
            next_index  = (key_index + 1) % len(GEMINI_API_KEYS)
            elapsed     = round(time.monotonic() - t0, 1)

            print(
                f"  [gemini] Rate-limited on key ...{api_key[-6:]} "
                f"(sweep hit {sweep_hits}/{len(GEMINI_API_KEYS)}, +{elapsed}s). "
                f"Rotating to key index {next_index}."
            )

            if sweep_hits >= len(GEMINI_API_KEYS):
                print(
                    f"  [gemini] All {len(GEMINI_API_KEYS)} key(s) exhausted in this sweep. "
                    f"Waiting {INTER_KEY_WAIT_S}s before retrying pool..."
                )
                time.sleep(INTER_KEY_WAIT_S)
                sweep_hits = 0

            key_index = next_index
            # Continue immediately to the next key.


# ---------------------------------------------------------------------------
# CONSTRAINTS
# ---------------------------------------------------------------------------

MIN_STANDOFF_M      = 5.0
VMAP_MIN_STANDOFF_M = 3.0
VMAP_MAX_STANDOFF_M = 7.0
VMAP_MIN_ALTITUDE_M = 2.0

_NO_REPLAN_AFTER: frozenset[str] = frozenset({"takeoff", "return_home"})

# ---------------------------------------------------------------------------
# PLANNING PROMPT  (identical to mission_executor.py)
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
                                                        standoff 3.0-7.0 m.
                                                        Use repeat: N for N full laps.
                                                        approach for the same
                                                        cylinder_id MUST precede this.

               min_altitude_m: float (m, >= 2.0)      Only for vertical_map mode.
                                                       Default 2.0.

return_home    (no args)                              Always the last step.

=== CONSTRAINTS ===

  1. First step MUST be takeoff. Last step must be return_home.
  2. approach(cylinder_id=X) MUST appear immediately before map(cylinder_id=X).
  3. Do not include approach or map unless the mission explicitly involves object
     interaction.
  4. Do not add steps that are not implied by the mission description.
  5. For mode "orbit": standoff_distance must be >= 5.0 m for both approach and map.
  6. For mode "vertical_map": standoff_distance must be between 3.0 m and 7.0 m.
  7. After search, assume objects may be found. The automatic replanner will
     adjust the plan based on what is actually discovered.

=== OUTPUT ===

Respond with a JSON array only. No explanation, no markdown, no backticks.
"""

# ---------------------------------------------------------------------------
# REPLAN PROMPT BUILDER  (mirrors mission_executor.py)
# ---------------------------------------------------------------------------

def build_replan_system_prompt(
    mission_intent:  str,
    completed_steps: list[dict],
    remaining_steps: list[dict],
    cylinders:       list["MockCylinder"],
    failure_context: str | None = None,
) -> str:
    """Build the full system prompt for the mid-mission replanner."""

    def _fmt_steps(steps, annotate_semantics=False):
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

    def _build_manifest(cyls):
        if not cyls:
            return "  No cylinders discovered yet."
        data = [
            {
                "cylinder_id":          c.id,
                "world_ned_x_m":        round(c.world_x, 2),
                "world_ned_y_m":        round(c.world_y, 2),
                "depth_at_detection_m": round(c.depth_m,  2),
            }
            for c in cyls
        ]
        return (
            f"  {len(cyls)} cylinder(s) discovered:\n"
            "  ```json\n"
            + "  " + json.dumps(data, indent=2).replace("\n", "\n  ")
            + "\n  ```"
        )

    mapped   = _extract_mapped_cylinders(completed_steps)
    if mapped:
        mapped_lines = [
            f"  - Cylinder {cid}: COMPLETE ({passes} pass(es) done)"
            for cid, passes in sorted(mapped.items())
        ]
        mapped_section = "\n".join(mapped_lines)
    else:
        mapped_section = "  (none yet)"

    unmapped_ids = [c.id for c in cylinders if c.id not in mapped]
    unmapped_note = (
        "  Cylinders NOT yet mapped: " + ", ".join(str(i) for i in unmapped_ids)
        if unmapped_ids else
        "  All discovered cylinders have been mapped."
    )

    failure_section = ""
    if failure_context:
        failure_section = f"""
=== SECTION 3b - FAILURE / ANOMALY REPORT ===

The following event occurred during execution that may require you to revise
the remaining plan:

  {failure_context}

Consider whether the remaining plan still makes sense given this event.
"""

    return f"""\
=== SECTION 1 - BACKGROUND ===

You are a mid-flight autonomous UAV mission replanner.

After each skill completes (or fails), you receive a status snapshot of the
ongoing mission. Your role is to review the remaining plan and decide whether
it still satisfies the user's original intent given what has been discovered
or has gone wrong so far. You may revise, extend, or trim the remaining steps
as needed. You do NOT re-plan from scratch - only the tail (steps yet to
execute) is yours to change. Completed steps are fixed history.
Do NOT make changes unless they are ABSOLUTELY NECESSARY to complete the
intended mission.


=== SECTION 2 - USER MISSION ===

  "{mission_intent}"

This is the exact mission the user requested. Use it as the ground truth
for what "success" means.


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
                                                     For vertical_map: 3.0-7.0 m

map           cylinder_id: int                       Two modes:
              standoff_distance: float (m)
              mode: "orbit"                          ORBIT: one full orbit per
                   | "vertical_map"                  invocation (standoff >= 5.0 m).
                                                     Use repeat: N to orbit N times.
              min_altitude_m: float (>= 2.0)         VERTICAL_MAP: systematic vertical
                                                     sweep (standoff 3.0-7.0 m).

return_home   (no args)                              Must be the last step.
{failure_section}

Required pattern for each cylinder you intend to map:
  {{"state":"approach","args":{{"standoff_distance":D,"cylinder_id":N}}}},
  {{"state":"map","args":{{"mode":"orbit","standoff_distance":D,"cylinder_id":N}},"repeat":K}}

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

MAPPING PROGRESS:
    Only completed "map" steps count as mapping passes.
{mapped_section}

{unmapped_note}

REMAINING PLAN (scheduled for execution):
{_fmt_steps(remaining_steps)}

DISCOVERED OBJECTS:
{_build_manifest(cylinders)}


=== SECTION 5 - OUTPUT DIRECTIVE ===

Do NOT repeat, echo, or summarise any part of this prompt in your response.

Carefully compare the remaining plan against the user mission.
If the remaining plan already fulfils the user mission, respond with exactly:

NOMINAL

ONLY if the remaining plan MUST be adjusted, output:
  1. A corrected JSON array of the complete revised tail (all steps from
     now until return_home). Every element must be a complete JSON object.
  2. Immediately after the JSON, on a new line starting with "REASON:", a
     single concise sentence explaining what was wrong and why your revision
     is necessary.

Do not include any other explanation, markdown, or backticks.
"""


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


# ---------------------------------------------------------------------------
# PLAN PARSING AND VALIDATION
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list:
    """Parse a JSON array from LLM output, repairing common model defects."""
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
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return [item for item in parsed if isinstance(item, dict)]


def validate_plan(plan: list, is_tail: bool = False) -> tuple[bool, list[str]]:
    """
    Returns (ok, errors).
    Validates structure, ordering, standoff distances, and constraint satisfaction.
    """
    errors: list[str] = []
    if not plan:
        return False, ["Plan is empty"]

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            errors.append(f"Step {i} is not a dict")

    if errors:
        return False, errors

    if not is_tail and plan[0].get("state") != "takeoff":
        errors.append("Must start with takeoff")
    if plan[-1].get("state") != "return_home":
        errors.append("Must end with return_home")

    valid_states = {"takeoff", "search", "approach", "map", "return_home"}
    approached: set[Any] = set()

    for i, step in enumerate(plan):
        s    = step.get("state", "")
        args = step.get("args", {})

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
            approached.add(args.get("cylinder_id"))
            if "standoff_distance" not in args:
                errors.append(f"approach at {i} missing standoff_distance")

        if s == "map":
            cid  = args.get("cylinder_id")
            sd   = float(args.get("standoff_distance", 0))
            mode = args.get("mode", "orbit")

            if cid not in approached and cid != "all":
                errors.append(
                    f"map at {i} (cylinder_id={cid}) requires a preceding approach")

            if mode not in ("orbit", "vertical_map"):
                errors.append(f"map at {i} invalid mode={mode!r}")

            if mode == "orbit" and sd < MIN_STANDOFF_M:
                errors.append(f"map(orbit) at {i} standoff={sd} < {MIN_STANDOFF_M}m")

            if mode == "vertical_map":
                if sd < VMAP_MIN_STANDOFF_M:
                    errors.append(f"map(vertical_map) at {i} standoff={sd} < {VMAP_MIN_STANDOFF_M}m")
                if sd > VMAP_MAX_STANDOFF_M:
                    errors.append(f"map(vertical_map) at {i} standoff={sd} > {VMAP_MAX_STANDOFF_M}m")
                min_alt = float(args.get("min_altitude_m", VMAP_MIN_ALTITUDE_M))
                if min_alt < VMAP_MIN_ALTITUDE_M:
                    errors.append(f"map(vertical_map) at {i} min_altitude_m={min_alt} < {VMAP_MIN_ALTITUDE_M}m")

            if "repeat" in args:
                errors.append(f"map at {i} has 'repeat' inside args (must be top-level)")

    return len(errors) == 0, errors


def parse_replan_response(raw: str) -> tuple[list | None, str | None]:
    """
    Parse replanner output.
    Returns (tail_plan, reason) or (None, None) for NOMINAL.
    """
    stripped = raw.strip()

    has_nominal   = bool(re.search(r'\bNOMINAL\b', stripped, re.IGNORECASE))
    has_json_plan = bool(re.search(r'\[\s*\{', stripped))

    if has_nominal and not has_json_plan:
        return None, None

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
    json_portion = stripped[best_start: best_end + 1]
    tail = extract_json(json_portion)
    return tail, reason


# ---------------------------------------------------------------------------
# MOCK WORLD STATE  (no ROS2 required)
# ---------------------------------------------------------------------------

@dataclass
class MockCylinder:
    id:      int
    world_x: float
    world_y: float
    depth_m: float


@dataclass
class MockWorldState:
    """
    Represents the state of the world at a given point in mission execution.
    Used to drive experiment scenarios without running Gazebo.
    """
    cylinders:        list[MockCylinder] = field(default_factory=list)
    completed_steps:  list[dict]         = field(default_factory=list)
    remaining_steps:  list[dict]         = field(default_factory=list)
    failure_context:  str | None         = None
    battery_pct:      float              = 100.0


def make_world(
    *,
    n_cylinders:     int   = 0,
    completed_steps: list  | None = None,
    remaining_steps: list  | None = None,
    failure_context: str   | None = None,
    battery_pct:     float = 100.0,
) -> MockWorldState:
    """Convenience factory for building MockWorldState in test scenarios."""
    cyls = [
        MockCylinder(id=i + 1, world_x=float(i * 5), world_y=0.0, depth_m=5.0)
        for i in range(n_cylinders)
    ]
    return MockWorldState(
        cylinders       = cyls,
        completed_steps = completed_steps or [],
        remaining_steps = remaining_steps or [],
        failure_context = failure_context,
        battery_pct     = battery_pct,
    )


# ---------------------------------------------------------------------------
# PLAN GENERATION (wraps query_llm + validate loop)
# ---------------------------------------------------------------------------

def generate_plan(mission: str, model_key: str, max_attempts: int = 3) -> dict:
    """
    Attempt to generate a valid plan for a mission.

    Each iteration of the loop represents one LOGICAL attempt (bad JSON,
    constraint violation, etc.).  Rate-limit retries inside query_llm are
    transparent and do NOT consume an attempt slot here.

    Returns a result dict with keys:
      success, plan, attempts, latency_s, raw_responses, errors
    """
    result = {
        "mission":       mission,
        "model":         model_key,
        "success":       False,
        "plan":          None,
        "attempts":      0,
        "latency_s":     0.0,
        "raw_responses": [],
        "errors":        [],
    }
    total_latency = 0.0

    for attempt in range(max_attempts):
        result["attempts"] += 1
        try:
            raw, latency = query_llm(mission, system=PLANNING_PROMPT, model_key=model_key)
            total_latency += latency
            result["raw_responses"].append(raw)
            plan = extract_json(raw)
            ok, errs = validate_plan(plan)
            if ok:
                result["success"]   = True
                result["plan"]      = plan
                result["latency_s"] = total_latency
                return result
            result["errors"].append(f"attempt {attempt+1}: " + "; ".join(errs))
        except Exception as exc:
            result["errors"].append(f"attempt {attempt+1}: {exc}")

    result["latency_s"] = total_latency
    return result


# ---------------------------------------------------------------------------
# REPLAN CALL (wraps query_llm + validate loop)
# ---------------------------------------------------------------------------

def run_replan(
    mission_intent:  str,
    world:           MockWorldState,
    model_key:       str,
    max_attempts:    int = 3,
) -> dict:
    """
    Call the replanner and return a result dict with keys:
      nominal, tail, reason, valid, attempts, latency_s, raw_responses, errors

    Rate-limit retries inside query_llm are transparent; only genuine
    model-logic failures (bad JSON, constraint violations) consume an attempt.
    """
    system_prompt = build_replan_system_prompt(
        mission_intent  = mission_intent,
        completed_steps = world.completed_steps,
        remaining_steps = world.remaining_steps,
        cylinders       = world.cylinders,
        failure_context = world.failure_context,
    )

    result = {
        "mission":       mission_intent,
        "model":         model_key,
        "nominal":       False,
        "tail":          None,
        "reason":        None,
        "valid":         False,
        "attempts":      0,
        "latency_s":     0.0,
        "raw_responses": [],
        "errors":        [],
    }
    total_latency = 0.0
    current_system = system_prompt

    for attempt in range(max_attempts):
        result["attempts"] += 1
        try:
            raw, latency = query_llm("", system=current_system, model_key=model_key)
            total_latency += latency
            result["raw_responses"].append(raw)

            tail, reason = parse_replan_response(raw)

            if tail is None:   # NOMINAL
                result["nominal"]   = True
                result["valid"]     = True
                result["latency_s"] = total_latency
                return result

            ok, errs = validate_plan(tail, is_tail=True)
            if ok:
                result["tail"]      = tail
                result["reason"]    = reason
                result["valid"]     = True
                result["latency_s"] = total_latency
                return result

            err_str = "; ".join(errs)
            result["errors"].append(f"attempt {attempt+1}: {err_str}")
            current_system = (
                current_system
                + f"\n\n--- PREVIOUS ATTEMPT {attempt+1} WAS INVALID ---\n"
                + "Errors found:\n"
                + "\n".join(f"  - {e}" for e in errs)
                + "\n\nFix all errors and output the corrected JSON array "
                  "followed by REASON: <one sentence>."
            )

        except Exception as exc:
            result["errors"].append(f"attempt {attempt+1}: {exc}")

    result["latency_s"] = total_latency
    return result


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------

def fmt_plan(plan: list) -> str:
    if not plan:
        return "  (empty)"
    lines = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            lines.append(f"  [{i}] <invalid: {step!r}>")
            continue
        args_str = ", ".join(f"{k}={v}" for k, v in step.get("args", {}).items())
        rep      = step.get("repeat", 1)
        rep_str  = f" x{rep}" if rep > 1 else ""
        lines.append(f"  [{i}] {step['state']:20} {args_str}{rep_str}")
    return "\n".join(lines)


def save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")