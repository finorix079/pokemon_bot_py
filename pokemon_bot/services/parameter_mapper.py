"""services/parameter_mapper.py — map model-supplied params onto API schema.

Ported from open-react-template/services/parameterMapper.ts.

Scoring rules, normalisation rules, and fan-out detection are all preserved
verbatim. The Chinese-comment console.log messages in the TS source are
kept (translated to f-strings) so log searches by message survive the port.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

ParamType = Literal[
    "number", "string", "boolean", "object", "array", "integer", "unknown"
]


@dataclass
class RequiredParam:
    name: str
    type: ParamType = "unknown"
    inPath: bool = False


@dataclass
class MappingResult:
    mapped: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)
    fanOutDetected: bool = False
    fanOutParam: Optional[str] = None
    fanOutValues: Optional[list[Any]] = None
    typeMismatchDetected: bool = False
    typeMismatchDetail: list[str] = field(default_factory=list)


_PATH_PARAM_RE = re.compile(r"\{(\w+)\}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_PLURAL_SUFFIX_RE = re.compile(r"(ids|idlist|list|array)$")
_DOMAIN_PREFIX_RE = re.compile(r"^(team|pokemon|user|admin|member|watch)")


def extract_path_params(path: str) -> list[str]:
    """Extract `{paramName}` placeholders from a URL path."""
    return _PATH_PARAM_RE.findall(path)


def norm_key(k: str) -> str:
    """Normalise a key for fuzzy matching.

    Rules (preserved from TS `normKey`):
    - lowercase
    - strip non-alnum
    - strip plural/list suffixes: `ids`, `idlist`, `list`, `array`
    - strip domain prefixes: `team`, `pokemon`, `user`, `admin`, `member`, `watch`
    - strip trailing `s` plural if length > 1
    """
    s = _NON_ALNUM_RE.sub("", k.lower())
    s = _PLURAL_SUFFIX_RE.sub("", s)
    s = _DOMAIN_PREFIX_RE.sub("", s)
    if s.endswith("s") and len(s) > 1:
        s = s[:-1]
    return s


def guess_value_type(v: Any) -> ParamType:
    """Guess a `ParamType` from a runtime value."""
    if isinstance(v, list):
        return "array"
    if v is None:
        return "unknown"
    if isinstance(v, bool):  # bool is subclass of int — check first
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, dict):
        return "object"
    return "unknown"


def extract_required_params(
    api_path: str,
    parameters: Optional[list[dict[str, Any]]] = None,
) -> list[RequiredParam]:
    """Extract required params from an OpenAPI path + parameters list.

    Path params (`/foo/{bar}`) are always treated as required and marked
    `inPath=True`. Other parameters are required only if `required=True`
    in the OpenAPI definition.
    """
    required: list[RequiredParam] = []

    for name in extract_path_params(api_path):
        param_schema = None
        if parameters:
            param_schema = next(
                (p for p in parameters if p.get("name") == name and p.get("in") == "path"),
                None,
            )
        ptype: ParamType = (
            (param_schema or {}).get("schema", {}).get("type") or "unknown"
        )
        required.append(RequiredParam(name=name, type=ptype, inPath=True))

    for param in parameters or []:
        if param.get("required") and param.get("in") != "path":
            ptype = (param.get("schema") or {}).get("type") or "unknown"
            required.append(RequiredParam(name=param["name"], type=ptype, inPath=False))

    return required


def map_args_to_required(
    required: list[RequiredParam],
    provided: dict[str, Any],
) -> MappingResult:
    """Map model-provided args to required params using exact + fuzzy match.

    Ported from TS `mapArgsToRequired`. Scoring rules:
    - Exact key match: takes the value directly (no scoring needed)
    - Normalised exact match: +10
    - Normalised partial match (either contains the other): +4
    - Type/structure match (number-vs-number, string-vs-string,
      array-of-correct-element-type): +4 or +3
    - `reqNorm == "id"` and key contains "id": +2
    - Path param weighting: +1
    - Threshold for accepting the best match: score >= 6

    Fan-out detection: path params receiving an array of length > 1 trigger
    `fanOutDetected=True` and surface the param/values for the caller.
    """
    mapped: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    fan_out_detected = False
    fan_out_param: Optional[str] = None
    fan_out_values: Optional[list[Any]] = None

    type_mismatch_detected = False
    type_mismatch_detail: list[str] = []

    provided_entries = list(provided.items())

    for req in required:
        # 1) Exact match
        if req.name in provided:
            value = provided[req.name]
            mapped[req.name] = value
            mapping[req.name] = req.name

            vt = guess_value_type(value)
            if req.type in ("number", "integer") and vt != "number":
                # Mirror the TS isNaN check via float() coercion.
                try:
                    float(value)
                except (TypeError, ValueError):
                    type_mismatch_detected = True
                    detail = (
                        f'参数 "{req.name}" 期望类型为 {req.type}，但实际为 {vt}'
                    )
                    type_mismatch_detail.append(detail)
                    print(
                        f'⚠️ 参数类型不匹配: "{req.name}" 期望 {req.type}，实际为 {vt}'
                    )
            if req.type == "string" and vt != "string":
                type_mismatch_detected = True
                detail = (
                    f'参数 "{req.name}" 期望类型为 string，但实际为 {vt}'
                )
                type_mismatch_detail.append(detail)
                print(
                    f'⚠️ 参数类型不匹配: "{req.name}" 期望 string，实际为 {vt}'
                )

            if req.inPath and isinstance(value, list) and len(value) > 1:
                fan_out_detected = True
                fan_out_param = req.name
                fan_out_values = list(value)
            continue

        # 2) Fuzzy match: normalised key + structure scoring
        req_norm = norm_key(req.name)
        best: Optional[dict[str, Any]] = None

        for k, v in provided_entries:
            k_norm = norm_key(k)
            score = 0

            if k_norm == req_norm:
                score += 10
            elif req_norm and (k_norm in req_norm or req_norm in k_norm):
                score += 4

            vt = guess_value_type(v)
            if req.type in ("number", "integer"):
                if vt == "number":
                    score += 4
                if vt == "array" and (len(v) == 0 or isinstance(v[0], (int, float))):
                    score += 3
            elif req.type == "string":
                if vt == "string":
                    score += 4
                if vt == "array" and (len(v) == 0 or isinstance(v[0], str)):
                    score += 3

            if req_norm == "id" and "id" in k.lower():
                score += 2

            if req.inPath:
                score += 1

            if best is None or score > best["score"]:
                best = {"key": k, "score": score}

        # 3) Apply best if it clears threshold
        if best and best["score"] >= 6:
            value = provided[best["key"]]
            mapped[req.name] = value
            mapping[req.name] = best["key"]

            vt = guess_value_type(value)
            if req.type in ("number", "integer") and vt != "number":
                type_mismatch_detected = True
                detail = (
                    f'参数 "{req.name}" 期望类型为 {req.type}，但实际为 {vt}'
                )
                type_mismatch_detail.append(detail)
                print(
                    f'⚠️ 参数类型不匹配: "{req.name}" 期望 {req.type}，实际为 {vt}'
                )
            if req.type == "string" and vt != "string":
                type_mismatch_detected = True
                detail = (
                    f'参数 "{req.name}" 期望类型为 string，但实际为 {vt}'
                )
                type_mismatch_detail.append(detail)
                print(
                    f'⚠️ 参数类型不匹配: "{req.name}" 期望 string，实际为 {vt}'
                )

            if req.inPath and isinstance(value, list) and len(value) > 1:
                fan_out_detected = True
                fan_out_param = req.name
                fan_out_values = list(value)

            print(
                f'✅ 参数映射: "{req.name}" <- "{best["key"]}" '
                f'(score: {best["score"]}{", path param" if req.inPath else ""})'
            )
        else:
            best_repr = (
                f' (最高分: {best["score"]}, key: {best["key"]})' if best else ""
            )
            print(f'⚠️  无法映射 required 参数 "{req.name}"{best_repr}')

    return MappingResult(
        mapped=mapped,
        mapping=mapping,
        fanOutDetected=fan_out_detected,
        fanOutParam=fan_out_param,
        fanOutValues=fan_out_values,
        typeMismatchDetected=type_mismatch_detected,
        typeMismatchDetail=type_mismatch_detail,
    )


def prepare_args_for_request(
    api_path: str,
    parameters: Optional[list[dict[str, Any]]],
    provided_args: dict[str, Any],
) -> MappingResult:
    """Convenience: extract required params + run the mapper."""
    required = extract_required_params(api_path, parameters)
    print("\n🔍 参数映射分析:")
    print(f"  API path: {api_path}")
    print(f"  Required params: {required}")
    print(f"  Provided args: {provided_args}")

    result = map_args_to_required(required, provided_args)
    print(f"  映射结果: {result.mapping}")
    if result.fanOutDetected:
        values = ", ".join(map(str, result.fanOutValues or []))
        print(f"  🔄 检测到 fan-out: {result.fanOutParam} = [{values}]")

    return result


__all__ = [
    "ParamType",
    "RequiredParam",
    "MappingResult",
    "extract_path_params",
    "norm_key",
    "guess_value_type",
    "extract_required_params",
    "map_args_to_required",
    "prepare_args_for_request",
]
