import os, re

with open('paper/Physics-Informed NoProp.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Check all referenced figure files exist
refs = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', content)
print('Figure file checks:')
for r in refs:
    candidate = os.path.join('paper', r)
    exists = os.path.exists(candidate)
    print(f'  {r:55s} {"OK" if exists else "MISSING"}')

# Check labels and eqrefs
labels = set(re.findall(r'\\label\{([^}]+)\}', content))
refs_set = set(re.findall(r'\\ref\{([^}]+)\}', content))
eqrefs = set(re.findall(r'\\eqref\{([^}]+)\}', content))

all_refs = refs_set | eqrefs
orphan = all_refs - labels
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
