"""3-pass JSON recovery for LLM outputs.

LLMs often produce "almost JSON." This module provides forgiving parsing
before spending a retry call.

Recovery order:
1. Extract JSON block (first { to last })
2. Light repair (trailing commas, smart quotes, etc.)
3. Parse
4. If still invalid: best-effort extraction so UI has something to show
"""

import json
import re
from typing import Any


def extract_json_block(text: str) -> str:
    """Extract JSON block from text (first { to matching }).

    Handles:
    - Markdown code blocks (```json ... ```)
    - Leading/trailing text
    - Nested braces
    """
    # Try markdown code block first
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    # Find first { and match to closing }
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    end = start
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    return text[start : end + 1]


def repair_json(text: str) -> str:
    """Light repair of common JSON issues.

    Fixes:
    - Trailing commas before } or ]
    - Smart quotes (""'') to straight quotes
    - Single quotes around keys/values (risky, best effort)
    - Unquoted keys (simple cases)
    """
    result = text

    # Smart quotes to straight quotes
    result = result.replace('"', '"').replace('"', '"')
    result = result.replace(""", "'").replace(""", "'")

    # Trailing commas before ] or }
    result = re.sub(r",(\s*[}\]])", r"\1", result)

    # Single quotes to double quotes (risky but often needed)
    # Only for simple cases where key/value pattern is clear
    # Pattern: 'key': or : 'value'
    result = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', result)
    result = re.sub(r"(:\s*)'([^']*)'", r'\1"\2"', result)

    # Unquoted keys (simple alphanumeric)
    result = re.sub(r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', result)

    return result


def parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON with 3-pass recovery.

    Returns:
        (parsed_dict, None) on success
        (None, error_message) on failure
    """
    # Pass 1: Try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Pass 2: Extract JSON block and try
    extracted = extract_json_block(text)
    try:
        return json.loads(extracted), None
    except json.JSONDecodeError:
        pass

    # Pass 3: Repair and try
    repaired = repair_json(extracted)
    try:
        return json.loads(repaired), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse failed after 3 passes: {e}"


def extract_best_effort_question(text: str) -> str | None:
    """Extract a question from text even if JSON parsing failed.

    Best-effort extraction so the UI has something to show.

    Tries:
    1. Look for "next_question" key and extract its value
    2. Find first line ending in ?
    3. Find any sentence ending in ?
    """
    # Try to find next_question value even in malformed JSON
    match = re.search(r'"?next_question"?\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)

    # Find first line ending in ?
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.endswith("?"):
            # Clean up any JSON artifacts
            cleaned = re.sub(r'^["\s{,]+', "", line)
            cleaned = re.sub(r'["\s},]+$', "", cleaned)
            if cleaned.endswith("?"):
                return cleaned

    # Find any sentence ending in ?
    sentences = re.findall(r"[^.!?]*\?", text)
    if sentences:
        cleaned = sentences[0].strip()
        cleaned = re.sub(r'^["\s{,]+', "", cleaned)
        return cleaned

    return None


def extract_best_effort_ledger_update(text: str) -> dict[str, Any]:
    """Extract ledger-like data from text even if JSON parsing failed.

    Returns partial ledger_update with whatever could be extracted.
    """
    result: dict[str, Any] = {}

    # Try to extract thesis
    thesis_match = re.search(r'"?thesis"?\s*:\s*"([^"]+)"', text)
    if thesis_match:
        result["thesis"] = thesis_match.group(1)

    # Try to extract inquiry
    inquiry_match = re.search(r'"?inquiry"?\s*:\s*"([^"]+)"', text)
    if inquiry_match:
        result["inquiry"] = inquiry_match.group(1)

    return result


def recover_socrates_turn(text: str) -> tuple[dict[str, Any], bool, str | None]:
    """Recover a Socrates turn response with best-effort fallback.

    Args:
        text: Raw LLM response text

    Returns:
        (output_dict, success, error_message)

        On success: (parsed_output, True, None)
        On failure: (best_effort_output, False, error_message)
    """
    parsed, error = parse_json(text)

    if parsed is not None:
        return parsed, True, None

    # Best-effort extraction
    question = extract_best_effort_question(text)
    ledger_update = extract_best_effort_ledger_update(text)

    best_effort = {
        "next_question": question or "[Could not extract question from response]",
        "question_type": "other",
        "question_type_detail": "parse_recovery",
        "why_this_question": ["Response parsing failed; extracted best-effort question"],
        "ledger_update": ledger_update,
        "stop_check": {"done": False, "criteria": []},
        "_parse_error": True,
        "_raw_response": text[:500],  # First 500 chars for debugging
    }

    return best_effort, False, error
