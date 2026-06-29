from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from air_agent.tools.registry import ToolRegistry


PLUGIN_MANIFEST_NAME = "air-agent-plugin.json"
_ALLOWED_CAPABILITIES = {"tools", "skills", "provider", "memory", "planner"}


class PluginLoadError(RuntimeError):
    pass


@dataclass(slots=True)
class PluginManifest:
    name: str
    version: str
    description: str
    entrypoint: str
    capabilities: list[str] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path = field(default_factory=Path)

    @classmethod
    def from_dir(cls, plugin_dir: str | Path) -> PluginManifest:
        path = Path(plugin_dir)
        manifest_path = path / PLUGIN_MANIFEST_NAME
        if not manifest_path.is_file():
            raise PluginLoadError(f"Plugin manifest not found: {manifest_path}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PluginLoadError(f"Failed to parse plugin manifest {manifest_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PluginLoadError("Plugin manifest must be a JSON object")

        for field_name in ["name", "version", "description", "entrypoint"]:
            value = raw.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise PluginLoadError(f"Plugin manifest missing required field: {field_name}")

        capabilities = raw.get("capabilities", [])
        if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
            raise PluginLoadError("Plugin manifest capabilities must be a list of strings")
        invalid = [capability for capability in capabilities if capability not in _ALLOWED_CAPABILITIES]
        if invalid:
            raise PluginLoadError(f"Invalid plugin capabilities: {', '.join(invalid)}")

        permissions = raw.get("permissions", {})
        if permissions is None:
            permissions = {}
        if not isinstance(permissions, dict):
            raise PluginLoadError("Plugin manifest permissions must be an object")

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise PluginLoadError("Plugin manifest metadata must be an object")

        return cls(
            name=raw["name"].strip(),
            version=raw["version"].strip(),
            description=raw["description"].strip(),
            entrypoint=raw["entrypoint"].strip(),
            capabilities=list(capabilities),
            permissions=dict(permissions),
            metadata=dict(metadata),
            path=path,
        )


@dataclass(slots=True)
class PluginContext:
    manifest: PluginManifest
    registry: ToolRegistry
    skills_dirs: list[str] = field(default_factory=list)
    provider: Any = None
    memory: Any = None
    planner: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_tool(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str = "",
        namespace: str | None = None,
    ) -> None:
        tool_name = name or func.__name__
        if namespace:
            tool_name = f"{namespace}.{tool_name}"
        try:
            self.registry.register(func, name=tool_name, description=description, conflict="error")
        except ValueError as exc:
            raise PluginLoadError(f"Plugin tool registration conflict for {tool_name}: {exc}") from exc

    def add_skills_dir(self, path: str) -> None:
        self.skills_dirs.append(path)

    def set_provider(self, provider: Any) -> None:
        self.provider = provider

    def set_memory(self, memory: Any) -> None:
        self.memory = memory

    def set_planner(self, planner: Any) -> None:
        self.planner = planner


@dataclass(slots=True)
class PluginLoadResult:
    manifest: PluginManifest
    context: PluginContext


def load_plugin(
    plugin_dir: str | Path,
    *,
    registry: ToolRegistry,
    plugin_permissions: dict[str, Any] | None,
) -> PluginLoadResult:
    manifest = PluginManifest.from_dir(plugin_dir)
    _authorize_manifest(manifest, plugin_permissions)
    entrypoint = _load_entrypoint(manifest)
    context = PluginContext(manifest=manifest, registry=registry)
    try:
        entrypoint(context)
    except PluginLoadError:
        raise
    except Exception as exc:
        raise PluginLoadError(f"Plugin {manifest.name} entrypoint failed: {exc}") from exc
    return PluginLoadResult(manifest=manifest, context=context)


def _authorize_manifest(
    manifest: PluginManifest,
    plugin_permissions: dict[str, Any] | None,
) -> None:
    if not manifest.permissions:
        return
    if not plugin_permissions or plugin_permissions.get(manifest.name) is not True:
        raise PluginLoadError(
            f"Plugin {manifest.name} declares permissions {manifest.permissions} "
            "but is not authorized in plugin_permissions"
        )


def _load_entrypoint(manifest: PluginManifest) -> Callable[[PluginContext], Any]:
    if ":" not in manifest.entrypoint:
        raise PluginLoadError(f"Plugin {manifest.name} entrypoint must use module:function format")
    module_name, function_name = manifest.entrypoint.split(":", 1)
    if not module_name or not function_name:
        raise PluginLoadError(f"Plugin {manifest.name} entrypoint must use module:function format")
    sys.path.insert(0, str(manifest.path))
    try:
        importlib.invalidate_caches()
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PluginLoadError(f"Failed to import plugin {manifest.name} entrypoint: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(manifest.path))
        except ValueError:
            pass
    entrypoint = getattr(module, function_name, None)
    if not callable(entrypoint):
        raise PluginLoadError(f"Plugin {manifest.name} entrypoint function not found: {function_name}")
    return entrypoint


__all__ = [
    "PLUGIN_MANIFEST_NAME",
    "PluginContext",
    "PluginLoadError",
    "PluginLoadResult",
    "PluginManifest",
    "load_plugin",
]
