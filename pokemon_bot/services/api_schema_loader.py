"""services/api_schema_loader.py — load OpenAPI JSON schemas for the planner.

Ported from open-react-template/services/apiSchemaLoader.ts.

The TS code reads schemas from `process.cwd()/src/doc/openapi-doc/*.json`.
For the Python CLI port we accept that directory either:
- via the env var `OPENAPI_SCHEMAS_DIR` (explicit override),
- via a `prompts/openapi-doc/` directory inside the package, or
- via the original repo path `../open-react-template/src/doc/openapi-doc/`
  (auto-discovered relative to this file).

If none of those exist, `load_open_api_schemas()` returns `{}` — the
planner still works without OpenAPI schemas because `dynamic_api_request`
falls back to "skip parameter mapping" in that case (`apiService.ts:58-60`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

_cached_schemas: Optional[dict[str, Any]] = None


def _candidate_dirs() -> list[Path]:
    """Resolve candidate directories holding OpenAPI JSON files."""
    candidates: list[Path] = []
    env_override = os.environ.get("OPENAPI_SCHEMAS_DIR")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    # Bundled prompts dir alongside the package
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / "prompts" / "openapi-doc")
    # Original repo location, relative to the project root
    project_root = here.parent
    candidates.append(project_root.parent / "open-react-template" / "src" / "doc" / "openapi-doc")
    return candidates


def load_open_api_schemas() -> dict[str, Any]:
    """Load and merge all OpenAPI schemas. Cached after the first call.

    Mirrors TS `loadOpenApiSchemas`: collects every `*.json` file's `paths`
    object into a single dict keyed by path string.
    """
    global _cached_schemas
    if _cached_schemas is not None:
        return _cached_schemas

    schemas: dict[str, Any] = {}
    schemas_dir = next((p for p in _candidate_dirs() if p.is_dir()), None)

    if schemas_dir is None:
        print(
            "⚠️  No OpenAPI schemas directory found "
            "(checked OPENAPI_SCHEMAS_DIR, ./pokemon_bot/prompts/openapi-doc, "
            "../open-react-template/src/doc/openapi-doc)"
        )
        _cached_schemas = {}
        return _cached_schemas

    try:
        for file_path in sorted(schemas_dir.glob("*.json")):
            try:
                with file_path.open(encoding="utf-8") as f:
                    schema = json.load(f)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Failed to parse {file_path}: {exc}")
                continue
            paths = schema.get("paths")
            if isinstance(paths, dict):
                schemas.update(paths)
        print(
            f"✅ 加载了 {len(schemas)} 个 API endpoints from OpenAPI schemas "
            f"({schemas_dir})"
        )
    except Exception as error:  # noqa: BLE001
        print(f"❌ 加载 OpenAPI schemas 失败: {error}")
        schemas = {}

    _cached_schemas = schemas
    return _cached_schemas


_PATH_PARAM_NORM_RE = re.compile(r"\{[^}]+\}")


def _normalise_path(path: str) -> str:
    """Replace every `{name}` segment with `{PARAM}` for fuzzy matching.

    Mirrors TS `path.replace(/\\{[^}]+\\}/g, '{PARAM}')`.
    """
    return _PATH_PARAM_NORM_RE.sub("{PARAM}", path)


def _resolve_path(schemas: dict[str, Any], api_path: str) -> Optional[str]:
    """Find a matching key in `schemas` for `api_path` (exact then fuzzy)."""
    if api_path in schemas:
        return api_path
    target_norm = _normalise_path(api_path)
    for schema_path in schemas:
        if _normalise_path(schema_path) == target_norm:
            return schema_path
    return None


def find_api_parameters(api_path: str, method: str) -> Optional[list[dict[str, Any]]]:
    """Find the OpenAPI `parameters` list for the given path + method.

    Ported from TS `findApiParameters`. Returns `None` if no match.
    """
    schemas = load_open_api_schemas()
    normalised_method = method.lower()

    matched_path = _resolve_path(schemas, api_path)
    if matched_path is None:
        print(f"⚠️  未找到匹配的 API schema: {api_path} {method}")
        return None

    method_schema = schemas[matched_path].get(normalised_method)
    if method_schema is None:
        print(f"⚠️  API {matched_path} 没有 {normalised_method} 方法")
        return None

    return method_schema.get("parameters")


def find_api_schema(api_path: str, method: str) -> Optional[dict[str, Any]]:
    """Return the full method schema (parameters + requestBody + responses + …).

    Ported from TS `findApiSchema`.
    """
    schemas = load_open_api_schemas()
    normalised_method = method.lower()
    matched_path = _resolve_path(schemas, api_path) or api_path
    return schemas.get(matched_path, {}).get(normalised_method)


__all__ = [
    "load_open_api_schemas",
    "find_api_parameters",
    "find_api_schema",
]
