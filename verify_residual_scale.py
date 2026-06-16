"""
Apply the paper's own scale-persistence test to its LAST surviving residual.

For N = 10^5 and 10^6, compute on the pooled complete-trajectory values:
  - full pooled per-digit deviation + TVD
  - a structureless scale-invariant (log-uniform) null over the SAME span -> its TVD
  - the map-specific GAP = full TVD - null TVD, and the per-digit d4/d6 signature
  - the excision curve (TVD over values >= 10^j), to re-test the funnel at each N

Question: does the map-specific residual (the d4 excess / d6 deficit, net of the
generic finite-span effect) DECAY as N grows tenfold? If yes, the last signature
is itself finite-range and closed. If it holds, it is a persistent map feature.

Memoized magnitude x leading-digit recurrence so 10^6 is feasible.
"""
import numpy as np, sys
from math import log10
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)
BEN = np.array([log10(1 + 1/d) for d in range(1, 10)])
def tvd(p): return 0.5 * np.abs(p - BEN).sum()
def devs(p): return "  ".join(f"{d+1}:{(p[d]-BEN[d])/BEN[d]*100:+5.1f}%" for d in range(9))

NDIG = 14            # values stay < 10^14 in this range
W = NDIG * 10        # flat index = ndig*10 + leading_digit

def idx_of(m):
    ld = m; nd = 1
    while ld >= 10:
        ld //= 10; nd += 1
    return nd * 10 + ld

def run(N):
    counts = np.zeros((N + 1, W), dtype=np.uint16)
    counts[1, idx_of(1)] = 1
    gmax = 1
    for n in range(2, N + 1):
        m = n; seg = []
        while m >= n:
            seg.append(idx_of(m))
            if m > gmax: gmax = m
            m = m // 2 if (m & 1) == 0 else 3 * m + 1
        row = counts[m].copy()        # m < n: full-trajectory hist already cached
        for i in seg:
            row[i] += 1
        counts[n] = row
    P = counts[1:N + 1].sum(axis=0, dtype=np.int64).reshape(NDIG, 10)
    digtot = P[:, 1:10].sum(axis=0).astype(float)
    full = digtot / digtot.sum()
    total = int(digtot.sum())
    # excision: keep values with ndig >= j+1  (i.e. >= 10^j)
    exc = []
    for j in range(0, 7):
        c = P[j + 1:, 1:10].sum(axis=0).astype(float)
        if c.sum() == 0: continue
        exc.append((j, tvd(c / c.sum()), c.sum() / total))
    # generic-span null over [1, gmax]
    M = 5_000_000
    mant = (rng.random(M) * log10(gmax)) % 1.0
    gp = np.bincount(np.floor(10 ** mant).astype(int), minlength=10)[1:10].astype(float)
    gp /= gp.sum()
    return dict(N=N, total=total, span=log10(gmax), full=full, ftvd=tvd(full),
                null=gp, ntvd=tvd(gp), exc=exc)

rows = []
for N in (1_000, 10_000, 100_000, 1_000_000):
    r = run(N)
    rows.append(r)
    print(f"\n===== N = {N:,}   pooled = {r['total']:,}   span = {r['span']:.2f} decades =====")
    print(f"  FULL        TVD = {r['ftvd']:.4f}   {devs(r['full'])}")
    print(f"  GENERIC NULL TVD = {r['ntvd']:.4f}")
    print(f"  MAP GAP = {r['ftvd'] - r['ntvd']:+.4f}   "
          f"d4 {(r['full'][3]-BEN[3])/BEN[3]*100:+.1f}%  d6 {(r['full'][5]-BEN[5])/BEN[5]*100:+.1f}%")
    print("  excision keep>=10^j: " + "  ".join(f"j{j}:{t:.4f}({f*100:.0f}%)" for j, t, f in r['exc']))

def mono_dec(seq):
    return all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))

Ns = [r['N'] for r in rows]
ftvd = [r['ftvd'] for r in rows]
ntvd = [r['ntvd'] for r in rows]
gap = [r['ftvd'] - r['ntvd'] for r in rows]
d4 = [(r['full'][3] - BEN[3]) / BEN[3] * 100 for r in rows]
d6a = [abs((r['full'][5] - BEN[5]) / BEN[5] * 100) for r in rows]
print("\n----- FOUR-POINT SCALE TREND (same N's as the headline Benford table) -----")
print("  N        : " + "  ".join(f"{n:>10,}" for n in Ns))
print("  full TVD : " + "  ".join(f"{x:>10.4f}" for x in ftvd) + f"   monotonic-decreasing: {mono_dec(ftvd)}")
print("  null TVD : " + "  ".join(f"{x:>10.4f}" for x in ntvd) + f"   monotonic-decreasing: {mono_dec(ntvd)}  (expected NO: boundary phase)")
print("  map gap  : " + "  ".join(f"{x:>+10.4f}" for x in gap) + f"   monotonic-decreasing: {mono_dec(gap)}")
print("  d4 excess: " + "  ".join(f"{x:>+9.1f}%" for x in d4) + f"   monotonic-decreasing: {mono_dec(d4)}")
print("  d6 |dev| : " + "  ".join(f"{x:>9.1f}%" for x in d6a) + f"   monotonic-decreasing: {mono_dec(d6a)}")
print("\nDone.")
