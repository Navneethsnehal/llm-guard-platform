"""Validate model output against a small JSON Schema subset."""

import json
import re
from dataclasses import dataclass
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SchemaDecision:
    decision: str
    reasons: tuple[str, ...]
    text: str


def check_schema(text: str, *, kind: str, schema: dict[str, Any] | None) -> SchemaDecision:
    if kind == "text" or kind == "":
        return SchemaDecision("allow", (), text)
    stripped = _FENCE.sub("", text.strip())
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        return SchemaDecision("block", ("invalid_json",), text)
    if kind == "json_object":
        if not isinstance(parsed, dict):
            return SchemaDecision("block", ("type_mismatch:$",), text)
        return SchemaDecision("allow", (), stripped)
    if kind != "json_schema":
        return SchemaDecision("block", ("unsupported_response_format",), text)
    if schema is None:
        return SchemaDecision("block", ("missing_schema",), text)
    errors = validate_instance(parsed, schema, "$")
    if errors:
        return SchemaDecision("block", tuple(errors), text)
    return SchemaDecision("allow", (), stripped)


def validate_instance(instance: object, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_ok(instance, expected):
        return [f"type_mismatch:{path}"]
    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        errors.append(f"enum:{path}")
    if expected == "object" and isinstance(instance, dict):
        errors.extend(_object_errors(instance, schema, path))
    if expected == "array" and isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                errors.extend(validate_instance(item, item_schema, f"{path}[{index}]"))
    if expected == "string" and isinstance(instance, str):
        errors.extend(_string_errors(instance, schema, path))
    if expected in {"number", "integer"} and isinstance(instance, (int, float)):
        errors.extend(_number_errors(instance, schema, path))
    return errors


def _object_errors(instance: dict[object, object], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = schema.get("required")
    properties = schema.get("properties")
    additional = schema.get("additionalProperties", True)
    prop_map = properties if isinstance(properties, dict) else {}
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in instance:
                errors.append(f"missing_required:{path}.{key}")
    for key, value in instance.items():
        if not isinstance(key, str):
            errors.append(f"type_mismatch:{path}")
            continue
        child = prop_map.get(key)
        if isinstance(child, dict):
            errors.extend(validate_instance(value, child, f"{path}.{key}"))
        elif additional is False:
            errors.append(f"additional_property:{path}.{key}")
    return errors


def _string_errors(instance: str, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if isinstance(minimum, int) and len(instance) < minimum:
        errors.append(f"min_length:{path}")
    if isinstance(maximum, int) and len(instance) > maximum:
        errors.append(f"max_length:{path}")
    return errors


def _number_errors(instance: int | float, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool) and instance < minimum:
        errors.append(f"minimum:{path}")
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and instance > maximum:
        errors.append(f"maximum:{path}")
    return errors


def _type_ok(instance: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return True
