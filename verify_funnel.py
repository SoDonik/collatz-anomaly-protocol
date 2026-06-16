"""
Dispositive test: is the pooled-Collatz Benford deviation driven by the shared
low-altitude FUNNEL, or is it just GENERIC finite-log-span deviation (which any
finite dataset shows and which ALSO decays ~1/log N)?

Three measurements, all on the SAME pooled object (every value of every complete
trajectory n=1..N, funnel included):

 (1) FULL pooled per-digit histogram + TVD.
 (2) EXCISION / bulk test: recompute TVD over values >= 10^j for j=0..5.
     - funnel hypothesis  => bulk -> Benford (TVD collapses); the big digit-4/6
       anomalies live in the small values.
     - generic-span hyp.  => bulk still spans ~all decades, so TVD barely moves.
 (3) GENERIC-SPAN NULL: a scale-invariant (log-uniform) sample over the SAME
     span [1, max]. This is the deviation a structureless finite-log-range source
     gives. If it's ~0.02 the deviation is generic; if ~0.001 it is NOT.
"""
import numpy as np, sys
from math import log10
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rng = np.random.default_rng(7)

BEN = np.array([log10(1 + 1/d) for d in range(1, 10)])
def tvd(p): return 0.5 * np.abs(p - BEN).sum()
def dev(p): return [f"{d+1}:{(p[d]-BEN[d])/BEN[d]*100:+.1f}" for d in range(9)]

N = 100_000
MAXD = 14
counts = np.zeros((MAXD, 10), dtype=np.int64)   # counts[ndigits][leading_digit]
gmax = 0
for start in range(1, N + 1):
    m = start
    while True:
        ld = m; nd = 1
        while ld >= 10:
            ld //= 10; nd += 1
        counts[nd][ld] += 1
        if m > gmax: gmax = m
        if m == 1: break
        m = m // 2 if (m & 1) == 0 else 3 * m + 1
total = int(counts.sum())

full = counts.sum(axis=0)[1:10].astype(float); full /= full.sum()
print(f"pooled total values: {total:,}   max value: {gmax:,}   span: {log10(gmax):.2f} decades")
print(f"FULL pooled TVD from Benford: {tvd(full):.4f}")
print(f"FULL per-digit dev %: {dev(full)}")
print()
print("EXCISION (recompute over values >= 10^j):")
for j in range(0, 6):
    c = counts[j+1:].sum(axis=0)[1:10].astype(float)
    if c.sum() == 0: continue
    p = c / c.sum()
    print(f"  keep >=10^{j}: {int(c.sum()):>10,} ({c.sum()/total*100:5.1f}% of pool)  "
          f"TVD={tvd(p):.4f}  d4={(p[3]-BEN[3])/BEN[3]*100:+5.1f}%  d6={(p[5]-BEN[5])/BEN[5]*100:+5.1f}%")
print()
M = 5_000_000
# generic-span null: scale-invariant over the SAME span [1, gmax]
mant = (rng.random(M) * log10(gmax)) % 1.0
ld = np.floor(10 ** mant).astype(int)
gp = np.bincount(ld, minlength=10)[1:10].astype(float); gp /= gp.sum()
print(f"GENERIC-SPAN NULL  (log-uniform over [1, {gmax:,}], {log10(gmax):.2f} decades):")
print(f"  TVD={tvd(gp):.4f}   per-digit dev %: {dev(gp)}")
# sanity: exactly 8 integer decades must be ~0 (pure Benford)
mant2 = (rng.random(M) * 8.0) % 1.0
gp2 = np.bincount(np.floor(10 ** mant2).astype(int), minlength=10)[1:10].astype(float); gp2 /= gp2.sum()
print(f"  (sanity) log-uniform over exactly 8 integer decades TVD={tvd(gp2):.4f}  <- should be ~0")
print("\nDone.")
