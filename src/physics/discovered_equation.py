"""Validated SPIDER equation artifacts used by the optimized trainer."""
from __future__ import annotations

import json
from pathlib import Path


SUPPORTED_TERMS = {
    'pressure_laplacian',
    'convection_divergence',
}


def load_validated_equation(path):
    """Load an equation artifact and reject unvalidated or unsupported output.

    Silent fallback from failed discovery to an analytic equation would break
    the provenance of the model, so all failures are explicit here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'SPIDER equation artifact not found: {path}')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    if artifact.get('schema_version') != 1:
        raise ValueError(f'Unsupported SPIDER artifact schema: {path}')
    validation = artifact.get('validation', {})
    if not validation.get('passed', False):
        reasons = '; '.join(validation.get('failure_reasons', []))
        raise ValueError(f'SPIDER equation did not pass validation: {reasons}')
    equation = artifact.get('equation', {})
    terms = equation.get('terms', [])
    coefficients = equation.get('coefficients', [])
    if len(terms) < 2 or len(terms) != len(coefficients):
        raise ValueError('SPIDER artifact has an invalid term/coefficient list')
    unsupported = set(terms) - SUPPORTED_TERMS
    if unsupported:
        raise ValueError(f'Optimized loss does not support discovered terms: {unsupported}')
    if set(terms) != SUPPORTED_TERMS:
        raise ValueError('Validated HIT equation must contain both PP terms')
    return artifact

