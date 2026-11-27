"""Reproduce both numerical blocks and empirical figures offline."""
from pathlib import Path
import json, subprocess, sys
root = Path(__file__).resolve().parent
for args in (['run.py'], ['insights/analyze.py', '.'], ['insights/plot.py']):
    subprocess.run([sys.executable, *args], cwd=root, check=True)
for rel in ('results/checks.json', 'insights/results/checks.json'):
    data = json.loads((root / rel).read_text())
    checks = data.get('checks', data)
    if not checks or not all(v is True for v in checks.values()):
        raise SystemExit(f'Failed checks: {rel}: {checks}')
    print(f'{rel}: {len(checks)}/{len(checks)} passed')
