"""services/type_style_service.py — Pokémon type → Tailwind class mapping.

Ported from open-react-template/services/typeStyleService.ts. Used only by
frontend pages in the TS project; preserved here for completeness of the
port even though the CLI never renders these styles.
"""

from __future__ import annotations

type_classes: dict[str, str] = {
    "fire": "bg-orange-500/15 text-orange-300 border border-orange-500/40",
    "water": "bg-blue-500/15 text-blue-300 border border-blue-500/40",
    "grass": "bg-green-500/15 text-green-300 border border-green-500/40",
    "electric": "bg-yellow-500/15 text-yellow-300 border border-yellow-500/40",
    "ice": "bg-cyan-400/15 text-cyan-200 border border-cyan-400/40",
    "fighting": "bg-red-600/15 text-red-300 border border-red-600/40",
    "poison": "bg-purple-500/15 text-purple-300 border border-purple-500/40",
    "ground": "bg-amber-600/15 text-amber-300 border border-amber-600/40",
    "flying": "bg-indigo-400/15 text-indigo-200 border border-indigo-400/40",
    "psychic": "bg-pink-500/15 text-pink-300 border border-pink-500/40",
    "bug": "bg-lime-500/15 text-lime-300 border border-lime-500/40",
    "rock": "bg-amber-800/15 text-amber-200 border border-amber-800/40",
    "ghost": "bg-indigo-700/15 text-indigo-200 border border-indigo-700/40",
    "dragon": "bg-purple-700/15 text-purple-200 border border-purple-700/40",
    "dark": "bg-gray-700/20 text-gray-200 border border-gray-700/50",
    "steel": "bg-slate-500/15 text-slate-200 border border-slate-500/40",
    "fairy": "bg-rose-400/15 text-rose-200 border border-rose-400/40",
    "normal": "bg-gray-500/15 text-gray-200 border border-gray-500/30",
    "stellar": "bg-amber-300/15 text-amber-100 border border-amber-300/40",
}


def get_type_class(type_name: str) -> str:
    key = (type_name or "normal").lower()
    return type_classes.get(key, type_classes["normal"])


__all__ = ["type_classes", "get_type_class"]
