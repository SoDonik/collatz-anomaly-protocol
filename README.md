# Vetting statistical anomalies in the Collatz map

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20723338.svg)](https://doi.org/10.5281/zenodo.20723338)

Reproducible code and manuscript for the paper *Vetting statistical anomalies in
the Collatz map: a falsification protocol and four worked failures*
(D. Kadirbekov, 2026).

The paper proposes a three-rule protocol for telling a genuine structural feature
of the Collatz map from an artifact of method (fix the null before measuring,
report effect sizes rather than significance at large pooled sample sizes, and
require any effect to persist rather than decay as the range grows), and applies
it to four candidate anomalies in trajectories of starting values up to 10^6, a
Benford "violation," a 2-adic valuation excess, a prime/composite stopping-time
gap, and a Sophie Germain ratio near 4/3. Each dissolves. The paper then applies
the same protocol to the authors' own surviving conjecture (a shared-funnel
explanation for the residual Benford deviation) and reports that it fails too.

Every table in the paper is reproducible from the scripts here.

## Contents

| File | Role |
|------|------|
| `collatz_protocol.tex`, `collatz_protocol.pdf` | the manuscript and its compiled PDF |
| `figures/` | the five vector figures |
| `01_generate_data.py` | writes `collatz_data.csv`, trajectory statistics for n = 1..10^6 |
| `verify_claims.py` | reproduces the four worked cases (Sections 3.2-3.5) |
| `verify_funnel.py` | the excision + generic-span-null test (Table 5; refutes the funnel mechanism) |
| `verify_residual_scale.py` | the four-point residual decomposition (Table 6; the monotonic per-digit decay) |

## Requirements

Python 3.10+ with NumPy and pandas:

```
pip install -r requirements.txt
```

Tested with Python 3.14, NumPy 2.4, pandas 3.0, on consumer hardware. The full
suite runs in minutes.

## Reproduce

```
# Cases 3 and 4 (primes, Sophie Germain) read a precomputed table of stopping times:
python 01_generate_data.py        # writes collatz_data.csv (~21 MB, a few minutes)

# The four worked cases (Sections 3.2-3.5):
python verify_claims.py           # add --full for the N = 10^6 reruns of Cases 1-2

# The capstone (Section 4):
python verify_funnel.py           # excision test, Table 5
python verify_residual_scale.py   # four-point signature, Table 6
```

`verify_funnel.py` and `verify_residual_scale.py` are self-contained: they
recompute trajectories with a memoized recurrence and need no CSV. Each script
prints the exact values that appear in the paper.

## Mapping scripts to paper tables

| Paper element | Script |
|---------------|--------|
| Tables 2-4 (Benford TVD, per-digit, v2 null), Cases 1-2 | `verify_claims.py` (`--full` for 10^6) |
| Cases 3-4 (prime/parity, Sophie Germain) | `verify_claims.py` (after `01_generate_data.py`) |
| Table 5 (excision test) | `verify_funnel.py` |
| Table 6 (four-point per-digit signature) | `verify_residual_scale.py` |

## Citation

```
D. Kadirbekov, Vetting statistical anomalies in the Collatz map: a falsification
protocol and four worked failures (version v1.0.1), Zenodo, 2026.
https://doi.org/10.5281/zenodo.20723338
```

## License

Code is released under the MIT License (see `LICENSE`). The manuscript text is
copyright 2026 Daniyal Kadirbekov.
