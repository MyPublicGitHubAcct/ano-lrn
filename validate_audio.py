"""Validate all Audio widgets across notebooks by executing cells and inspecting signals."""
import json
import sys
import re
import io
import contextlib
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

# ── mock IPython ──────────────────────────────────────────────────────────────
class _Audio:
    instances: list = []

    def __init__(self, data, rate=None, **kwargs):
        _Audio.instances.append({'data': np.array(data, dtype=float), 'rate': rate})

def _display(*args, **kwargs):
    pass

import types
_ipython_mod = types.ModuleType('IPython')
_display_mod = types.ModuleType('IPython.display')
_display_mod.Audio = _Audio
_display_mod.display = _display
_ipython_mod.display = _display_mod
_ipython_mod.version_info = (8, 0, 0, 'final', 0)
_ipython_mod.get_ipython = lambda: None
sys.modules['IPython'] = _ipython_mod
sys.modules['IPython.display'] = _display_mod
sys.modules['IPython.core'] = types.ModuleType('IPython.core')
sys.modules['IPython.core.interactiveshell'] = types.ModuleType('IPython.core.interactiveshell')

# ── helpers ───────────────────────────────────────────────────────────────────
SPARSE_THRESHOLD = 0.001   # flag if <0.1% of samples are non-zero
PEAK_THRESHOLD   = 0.005   # flag if peak amplitude is below this

def clean_src(src):
    """Strip IPython magic and redirect path so cells run from repo root."""
    lines = []
    for line in src.split('\n'):
        s = line.strip()
        if s.startswith('%') or s.startswith('!'):
            continue
        # notebooks use '../src'; we run from repo root with 'src' already on path
        line = line.replace("sys.path.insert(0, '../src')", "pass")
        # replace inline imports of Audio/display with our mocks
        if re.match(r'\s*from IPython\.display import', line):
            line = re.sub(
                r'from IPython\.display import.*',
                'from IPython.display import Audio, display',
                line,
            )
        lines.append(line)
    return '\n'.join(lines)


def validate_audio(data: np.ndarray, idx: int) -> list[str]:
    issues = []
    if np.any(np.isnan(data)):
        issues.append(f'  Audio {idx}: contains NaN')
        return issues
    if np.any(np.isinf(data)):
        issues.append(f'  Audio {idx}: contains Inf')
        return issues
    peak = float(np.max(np.abs(data)))
    if peak < PEAK_THRESHOLD:
        issues.append(f'  Audio {idx}: peak too low ({peak:.6f})')
        return issues
    nonzero = int(np.count_nonzero(np.abs(data) > 1e-10))
    total = data.size
    ratio = nonzero / total
    if ratio < SPARSE_THRESHOLD:
        issues.append(
            f'  Audio {idx}: sparse signal — {nonzero}/{total} non-zero '
            f'({ratio*100:.3f}%) — likely single-sample impulse'
        )
    return issues


def run_notebook(nb_path: Path):
    _Audio.instances = []
    plt.close('all')

    with open(nb_path) as f:
        nb = json.load(f)

    ctx: dict = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
        'Audio': _Audio,
        'display': _display,
    }

    cell_errors = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = clean_src(''.join(cell['source']))
        if not src.strip():
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, str(nb_path), 'exec'), ctx)  # noqa: S102
        except Exception as e:
            cell_errors.append(f'  Cell {i}: {type(e).__name__}: {e}')

    audio_issues = []
    for idx, inst in enumerate(_Audio.instances, 1):
        audio_issues.extend(validate_audio(inst['data'], idx))

    return {
        'audio_count': len(_Audio.instances),
        'cell_errors': cell_errors,
        'audio_issues': audio_issues,
    }


# ── main ──────────────────────────────────────────────────────────────────────
notebooks = sorted(Path('notebooks').glob('*.ipynb'))

ok, warn = [], []

for nb_path in notebooks:
    try:
        result = run_notebook(nb_path)
    except Exception as e:
        warn.append((nb_path.name, [f'  FATAL: {e}'], [], 0))
        continue

    all_issues = result['cell_errors'] + result['audio_issues']
    if all_issues:
        warn.append((nb_path.name, result['cell_errors'], result['audio_issues'], result['audio_count']))
    else:
        ok.append((nb_path.name, result['audio_count']))

print('=== OK ===')
for name, n in ok:
    marker = '' if n > 0 else '  (no Audio widgets)'
    print(f'  {name} [{n} widget(s)]{marker}')

print()
print('=== ISSUES ===')
if not warn:
    print('  none')
for name, errs, audio_issues, n in warn:
    print(f'\n  {name} [{n} widget(s)]')
    for e in errs:
        print(e)
    for a in audio_issues:
        print(a)
