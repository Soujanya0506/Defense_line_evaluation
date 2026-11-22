"""Checks that two output folders contain the same numbers (e.g. the original notebook's outputs and
this repository's outputs).

    python scripts/compare_outputs.py --old /path/to/notebook_outputs --new outputs

Every CSV present in both folders is compared column by column (numbers with a relative tolerance of 1e-9;
1e-5 for robust_mixed_model.csv, whose mixed-model estimates come from an iterative optimizer).
features_corrected.csv (notebook) is compared with features.csv (this repository).
"""
import argparse
import os

import numpy as np
import pandas as pd

ALIASES = {'features.csv': 'features_corrected.csv'}
RTOL = {'robust_mixed_model.csv': 1e-5}


def compare(a, b, rtol=1e-9):
    """'' if equal, otherwise a short description of the first difference."""
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    for c in a.columns:
        x, y = a[c], b[c]
        xn, yn = pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')
        if xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum() and x.notna().any():
            if not np.allclose(xn.to_numpy(float), yn.to_numpy(float), rtol=rtol, atol=1e-8 if rtol > 1e-9 else 1e-12,
                               equal_nan=True):
                i = int(np.flatnonzero(~np.isclose(xn, yn, rtol=rtol, atol=1e-8 if rtol > 1e-9 else 1e-12, equal_nan=True))[0])
                return f"column '{c}', row {i}: {x.iloc[i]} vs {y.iloc[i]}"
        elif not x.astype(str).equals(y.astype(str)):
            i = int(np.flatnonzero(x.astype(str).to_numpy() != y.astype(str).to_numpy())[0])
            return f"column '{c}', row {i}: {x.iloc[i]} vs {y.iloc[i]}"
    return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    args = ap.parse_args()
    n_same = n_diff = 0
    for name in sorted(f for f in os.listdir(args.new) if f.endswith('.csv')):
        old_name = name if os.path.exists(os.path.join(args.old, name)) else ALIASES.get(name, name)
        old_path = os.path.join(args.old, old_name)
        if not os.path.exists(old_path):
            print(f"  (not in old folder)  {name}")
            continue
        res = compare(pd.read_csv(old_path), pd.read_csv(os.path.join(args.new, name)), RTOL.get(name, 1e-9))
        print(f"{'SAME' if not res else 'DIFF'}  {name}" + (f": {res}" if res else ''))
        n_same += not res
        n_diff += bool(res)
    print(f"\n{n_same} files identical, {n_diff} different")


if __name__ == '__main__':
    main()
