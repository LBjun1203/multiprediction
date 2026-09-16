"""Canonical CT ingestion and representation builders."""

from .manifest import LEGACY_CT_INPUT_SPEC, CTInputSpec, resolve_ct_input_spec

__all__ = ["LEGACY_CT_INPUT_SPEC", "CTInputSpec", "resolve_ct_input_spec"]
