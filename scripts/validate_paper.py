import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with open('paper/Physics-Informed NoProp.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Guard against restoring superseded single-snapshot claims.
forbidden = ('0.2929985', '60 archived', '80 archived',
             'PI-NoProp (SPIDER)',
             'Dataset: Johns Hopkins Turbulence Database')
present = [token for token in forbidden if token.lower() in content.lower()]
assert not present, f'Manuscript retains superseded content: {present}'
required = ('1.003459', '0.00501841', 'trajectory-bootstrap',
            'fig_spider_noise.pdf', 'fig_framework.pdf', 'fig_latent_metrics.pdf',
            r'\input{experiment_values.tex}', 'eq:context-gate',
            r'$32^3$', r'$16^3$', r'$\lambda=0.1$')
missing = [token for token in required if token not in content]
assert not missing, f'Manuscript is missing current evidence: {missing}'

# Numerical values must be current, protocol-matched, and source-hashed.
subprocess.run([sys.executable, 'scripts/sync_paper_results.py', '--check'],
               check=True)
values = json.loads(Path(
    'outputs/aggregate/full_ns_v5_input32_target16_residual_warmstart_paper_evidence.json'
).read_text(encoding='utf-8'))['values']
for macro in re.findall(r'\\([A-Za-z]+)', content):
    if re.match(r'^(?:low|high|lambda(?:Zero|Tiny|Small|Standard|Medium|Selected|Accuracy)'
                r'|relation(?:NS|PP|Full)|decoder(?:Linear|Conv|Parameter))', macro):
        assert macro in values, f'Undefined experiment value: {macro}'

abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
                     content, re.S).group(1).strip()
assert not re.search(r'\n[ \t]*\n', abstract), 'Abstract must remain one paragraph'

active = re.sub(r'(?m)^\s*%.*$', '', content)
cited = {key.strip()
         for entry in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', active)
         for key in entry.split(',')}
bibliography = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', active))
assert cited <= bibliography, f'Missing bibliography entries: {cited-bibliography}'
assert bibliography <= cited, f'Uncited bibliography entries: {bibliography-cited}'
print(f'Citation keys checked: {len(cited)}')

# Check all referenced figure files exist
refs = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', content)
from scripts.plot_results import PAPER_FIGURES, FIGURE_CODE
expected_figures = [f'figures/{name}.pdf' for name, _ in PAPER_FIGURES]
assert refs == expected_figures, f'Figure inventory/order mismatch: {refs}'
for name, _ in PAPER_FIGURES:
    assert (FIGURE_CODE / f'{name}.py').is_file(), f'Missing matching figure code: {name}'
print('Figure file checks:')
for r in refs:
    candidate = os.path.join('paper', r)
    exists = os.path.exists(candidate)
    print(f'  {r:55s} {"OK" if exists else "MISSING"}')
    assert exists, f'Missing figure: {candidate}'

# Check labels and eqrefs
label_list = re.findall(r'\\label\{([^}]+)\}', content)
labels = set(label_list)
duplicates = sorted(label for label in labels if label_list.count(label) > 1)
assert not duplicates, f'Duplicate labels: {duplicates}'
refs_set = set(re.findall(r'\\ref\{([^}]+)\}', content))
eqrefs = set(re.findall(r'\\eqref\{([^}]+)\}', content))

all_refs = refs_set | eqrefs
orphan = all_refs - labels
assert not orphan, f'Orphan references: {orphan}'
print(f'\nOrphan refs: {orphan if orphan else "None"}')

figure_count = len(re.findall(r'\\begin\{figure', content))
table_count = len(re.findall(r'\\begin\{table', content))
equation_count = len(re.findall(r'\\begin\{equation', content))
print(f'\nTotal figures: {figure_count}')
print(f'Total tables: {table_count}')
print(f'Total equations: {equation_count}')
print(f'Total labels: {len(labels)}')
print(f'Total refs: {len(all_refs)}')
print(f'Total includegraphics: {len(refs)}')
