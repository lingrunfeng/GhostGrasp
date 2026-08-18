"""Shared live tabletop geometry helpers for simulation and real D435 pipelines."""

from ghost_mgg_core_py.live_tabletop.components import extract_components
from ghost_mgg_core_py.live_tabletop.evidence import build_tabletop_evidence
from ghost_mgg_core_py.live_tabletop.fitters import fit_component_geometry
from ghost_mgg_core_py.live_tabletop.grouping import (
    ComponentGroupingDecision,
    ComponentGroupingResult,
    group_components_by_foreground_islands,
)
from ghost_mgg_core_py.live_tabletop.ranking import rank_tabletop_components, score_geometry_fit
from ghost_mgg_core_py.live_tabletop.types import (
    GeometryFit,
    PixelComponent,
    RankedGeometryHypothesis,
    TabletopEvidence,
)

__all__ = [
    "ComponentGroupingDecision",
    "ComponentGroupingResult",
    "GeometryFit",
    "PixelComponent",
    "RankedGeometryHypothesis",
    "TabletopEvidence",
    "build_tabletop_evidence",
    "extract_components",
    "fit_component_geometry",
    "group_components_by_foreground_islands",
    "rank_tabletop_components",
    "score_geometry_fit",
]
