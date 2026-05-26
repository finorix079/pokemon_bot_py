"""services/api_service.py — dynamic API request executor.

Ported from open-react-template/services/apiService.ts.

Replaces axios with httpx.AsyncClient. The TS code's parameter-mapping,
fan-out detection, default PokéAPI list limit, and request-config logging
are all preserved.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import quote, urlencode

import httpx

from .parameter_mapper import prepare_args_for_request, MappingResult


_LIST_ENDPOINT_RE = __import__("re").compile(r"^/(pokemon|move|ability|berry)/?$")


class FanOutRequest(dict):
    """Marker dict subclass returned when the mapper detects an array passed
    to a scalar path-param. The caller is expected to iterate fan-out values
    and re-issue per-value requests.

    Mirrors TS `FanOutRequest` interface. Subclassing `dict` keeps it
    JSON-serialisable while still being distinguishable via `isinstance`.
    """


async def dynamic_api_request(base_url: str, schema: dict[str, Any]) -> Any:
    """Dynamically build and send an API request from a JSON schema.

    Ported from TS `dynamicApiRequest`. Key behaviours preserved:
    - Auto parameter mapping (model-supplied → schema-required key names)
    - Type-mismatch detection raises `RuntimeError` with `❌ 参数类型不匹配:` prefix
    - Fan-out detection returns a `FanOutRequest` instead of executing
    - Default `limit=20` for `/pokemon`, `/move`, `/ability`, `/berry` list endpoints
    - Request body forwarded as JSON for non-GET methods
    """
    try:
        print(f"Dynamic API Request Schema: {schema}")
        path: str = schema["path"]
        method: str = schema["method"]
        request_body: Optional[dict[str, Any]] = schema.get("requestBody")
        provided_args: dict[str, Any] = (
            schema.get("parameters") or schema.get("input") or {}
        )

        mapping_result: Optional[MappingResult] = None
        path_params: dict[str, Any] = provided_args

        api_parameters = schema.get("parametersSchema") or schema.get("apiParameters")
        if api_parameters is not None:
            mapping_result = prepare_args_for_request(path, api_parameters, provided_args)
            path_params = mapping_result.mapped

            if mapping_result.typeMismatchDetected:
                msg = (
                    f"❌ 参数类型不匹配: "
                    f"{'; '.join(mapping_result.typeMismatchDetail or [])}"
                )
                print(msg)
                raise RuntimeError(msg)

            if (
                mapping_result.fanOutDetected
                and mapping_result.fanOutParam is not None
                and mapping_result.fanOutValues is not None
            ):
                print("🔄 检测到 fan-out 需求，返回 FanOutRequest")
                return FanOutRequest(
                    needsFanOut=True,
                    fanOutParam=mapping_result.fanOutParam,
                    fanOutValues=mapping_result.fanOutValues,
                    baseSchema=schema,
                    mappedParams=path_params,
                )
        else:
            print("⚠️  Schema 未提供 parametersSchema，跳过参数映射")

        # Path parameter substitution / query string accumulation
        final_path = path
        query_params: dict[str, str] = {}

        print("Path parameter replacement:")
        print(f"  - Original path: {path}")
        print(f"  - Original parameters: {json.dumps(provided_args, default=str)}")
        print(f"  - Mapped pathParams: {json.dumps(path_params, default=str)}")
        if mapping_result is not None:
            print(f"  - Mapping: {json.dumps(mapping_result.mapping)}")

        if isinstance(path_params, dict):
            for key, value in path_params.items():
                placeholder = "{" + key + "}"
                if placeholder in final_path:
                    final_path = final_path.replace(placeholder, quote(str(value)))
                    print(f"  ✅ Replaced {placeholder} with {value} in path")
                elif value is not None and value != "":
                    query_params[key] = str(value)

        # PokéAPI list endpoints get a default page limit so responses stay
        # within token budgets (mirrors apiService.ts:88-93).
        is_list_endpoint = bool(_LIST_ENDPOINT_RE.match(final_path))
        if method.lower() == "get" and is_list_endpoint and "limit" not in query_params:
            query_params["limit"] = "20"
            print("  ℹ️  Applied default limit=20 for PokéAPI list endpoint")

        query_string = "?" + urlencode(query_params) if query_params else ""
        print(f"  - Final path: {final_path}{query_string}")

        config: dict[str, Any] = {
            "method": method.lower(),
            "url": f"{base_url}{final_path}{query_string}",
            "headers": {"Content-Type": "application/json"},
        }
        if request_body and method.lower() != "get":
            config["json"] = request_body

        print(f"Dynamic API Request Config: {json.dumps(config, indent=2, default=str)}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(**config)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text
    except Exception as error:  # noqa: BLE001
        print(f"Error in dynamicApiRequest: {error}")
        raise


__all__ = ["dynamic_api_request", "FanOutRequest"]
