"""Record which physical statements are discovered, prescribed, or derived."""
import json
import re
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    vector = load_json('outputs/spider/vector_equation.json')
    scalar = load_json('outputs/spider/scalar_equation.json')
    spatial = load_json('outputs/spider/hit_spatial_equation.json')
    log = Path('data/generated/dns_log.txt').read_text(encoding='utf-8')
    header = re.search(r'N=(\d+)\^3\s+nu=([\d.]+)\s+dt=([\d.]+)', log)
    final = re.search(r'Final: Re_lambda=([\d.]+), u_rms=([\d.]+)', log)

    vector_terms = set(vector.get('active_terms', []))
    scalar_terms = set(scalar.get('active_terms', []))
    canonical_ns = {'time_deriv', 'convection', 'gradient', 'laplacian'}
    direct_pp = {'scalar_laplacian', 'convection_divergence'}
    report = {
        'dataset_facts': {
            'source': 'in-house forced HIT DNS',
            'grid': int(header.group(1)) if header else None,
            'viscosity': float(header.group(2)) if header else None,
            'solver_dt': float(header.group(3)) if header else None,
            'snapshot_stride_steps': 50,
            'snapshot_dt': (float(header.group(3)) * 50) if header else None,
            'n_snapshots': 80,
            'subdomain_shape': [32, 32, 32],
            'frames_per_npz': 32,
            're_lambda_final': float(final.group(1)) if final else None,
            'u_rms_final': float(final.group(2)) if final else None,
        },
        'legacy_spider_saved_output': {
            'vector_terms': vector.get('active_terms', []),
            'vector_coefficients': vector.get('coefficients', []),
            'scalar_terms': scalar.get('active_terms', []),
            'scalar_coefficients': scalar.get('coefficients', []),
            'direct_full_ns_recovery': canonical_ns.issubset(vector_terms),
            'direct_pressure_poisson_recovery': direct_pp.issubset(scalar_terms),
        },
        'validated_spatial_spider': {
            'terms': spatial['equation']['terms'],
            'coefficients': spatial['equation']['coefficients'],
            'validation_passed': spatial['validation']['passed'],
            'validation_eta': spatial['metrics']['validation_eta'],
            'bootstrap_support_fraction': spatial['metrics']['bootstrap_support_fraction'],
            'direct_pressure_poisson_structure_recovery': (
                set(spatial['equation']['terms']) ==
                {'pressure_laplacian', 'convection_divergence'}),
            'used_by_optimized_training': True,
        },
        'constraint_provenance': {
            'continuity': 'prescribed from incompressible HIT solver',
            'navier_stokes': 'prescribed from the DNS governing equations',
            'pressure_poisson_analytic_baseline': 'analytically derived from Navier-Stokes + continuity',
            'pressure_poisson_discovered': 'selected and fitted by spatial weak-form SPIDER on 60 snapshots; validated on 20 disjoint snapshots',
            'continuity_in_training': 'prescribed; not claimed as a new discovery',
        },
        'paper_guardrails': [
            'Do not state that the saved SPIDER output directly recovered full Navier-Stokes.',
            'Distinguish the failed legacy broad-library output from the validated spatial SPIDER artifact.',
            'State that spatial SPIDER recovered PP structure and an effective coefficient; do not claim full Navier-Stokes recovery.',
            'Report Re_lambda=30.5 for the generated dataset, not 55--140.',
            'Report solver dt=0.001 and snapshot spacing 0.05.',
        ],
    }
    output = Path('outputs/aggregate/physics_provenance.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
