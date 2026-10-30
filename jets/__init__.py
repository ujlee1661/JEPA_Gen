"""JETS: joint-embedding temporal sequence generation."""
from .models import JETSGenerator
from .encoders import TimeTargetEncoder, FeatureTargetEncoder

TimeTeacher = TimeTargetEncoder
FeatureTeacher = FeatureTargetEncoder

__all__ = ["JETSGenerator", "TimeTargetEncoder", "FeatureTargetEncoder"]
