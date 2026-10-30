"""Backward-compatible aliases; use :mod:`jets.encoders` for new code."""
from .encoders import TimeTargetEncoder as TimeTeacher
from .encoders import FeatureTargetEncoder as FeatureTeacher

__all__ = ["TimeTeacher", "FeatureTeacher"]
