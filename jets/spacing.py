"""Compatibility imports; new code should use :mod:`jets.layers.spacing`."""
from .layers.spacing import (decode_spacing, encode_spacing, face_log_mass,
                             first_active_log_probs, project_clr, sample_face)

__all__ = ["decode_spacing", "encode_spacing", "face_log_mass", "first_active_log_probs", "project_clr", "sample_face"]
