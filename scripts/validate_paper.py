import os, re

with open('paper/Physics-Informed NoProp.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Guard against restoring claims from the archived single-snapshot v1 study.
forbidden = ('0.2929985', '60 archived', '80 archived',
             'PI-NoProp (SPIDER)', 'fig_spider_discovery.png',
             'Dataset: Johns Hopkins Turbulence Database')
present = [token for token in forbidden if token.lower() in content.lower()]
assert not present, f'Manuscript retains superseded v1 content: {present}'
required = ('1.003459', '0.00501841', 'trajectory-bootstrap',
            'fig_v2_spider.png', 'fig_v2_main_results.png')
missing = [token for token in required if token not in content]
assert not missing, f'Manuscript is missing v2 evidence: {missing}'

# Check all referenced figure files exist
refs = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', content)
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
