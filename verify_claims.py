"""
Rigor audit of the paper's headline claims.

  CLAIM A: "v_2 = 4 occurs 42% more often than the geometric prediction after 3n+1"
  CLAIM B: "Collatz trajectories violate Benford's Law (chi^2 = 61,154), digit 3 down 6%"
  CLAIM C: "Primes have 4.1% longer trajectories than composites"        (second pass)
  CLAIM D: "Sophie Germain ratio ST(2p+1)/ST(p) ~ 4/3"                   (second pass)

A and B compare an empirical distribution to a null without checking whether the
null is the right one. C and D are tested against the obvious confounds: parity
and size for C (primes are odd; composites are mostly even), estimator choice
and a no-primality control for D (mean-of-ratios is inflated by tiny
denominators at small p).

Run:        python verify_claims.py            (claims A-D at default scale)
            python verify_claims.py --full     (adds the N = 10^6 reruns of A and B
                                                cited in the paper; a few extra minutes)

Claims C and D read total stopping times from collatz_data.csv (built by
01_generate_data.py) instead of recomputing 10^6 trajectories.
"""
import argparse
import math
import os
import random
from collections import Counter

random.seed(12345)  # reproducible

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'collatz_data.csv')


# ----------------------------------------------------------------------------- #
# Shared Collatz primitives
# ----------------------------------------------------------------------------- #
def v2(n):
    """2-adic valuation: number of times 2 divides n."""
    v = 0
    while n % 2 == 0:
        v += 1
        n //= 2
    return v


def leading_digit(n):
    while n >= 10:
        n //= 10
    return n


def load_stopping_times():
    """total_stopping_time indexed by n, from collatz_data.csv, plus a prime sieve."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    n = df['n'].to_numpy()
    n_max = int(n.max())
    st = np.zeros(n_max + 1, dtype=np.int64)
    st[n] = df['total_stopping_time'].to_numpy()
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n_max ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return st, sieve, n_max


# ============================================================================= #
# CLAIM A: the v_2 = 4 "excess"
# ============================================================================= #
def v2_null_uniform_odds(n_samples):
    """
    The CORRECT null for "v_2(3n+1) over odd n".

    For a UNIFORM random odd n, v_2(3n+1) is provably geometric: P(k) = 2^-k.
    We confirm this empirically so the baseline is not in dispute.
    """
    counts = Counter()
    for _ in range(n_samples):
        n = random.randrange(1, 10**9, 2)  # uniform random odd
        counts[v2(3 * n + 1)] += 1
    return counts


def v2_trajectory_visited(max_n):
    """
    What the ORIGINAL script measured: v_2(3n+1) at every odd step encountered
    while running Collatz on starting values 1..max_n. These odd numbers are a
    biased subset of all odds (they are the ones trajectories actually visit),
    so deviation from geometric can come from sampling, not from 3n+1 structure.

    Memoized recurrence: the v_2 counts of n's trajectory equal the counts
    recorded until the trajectory first drops below n, plus the cached counts of
    that smaller value. This makes max_n = 10^6 feasible.
    """
    import numpy as np
    KMAX = 40  # values stay below 2^40 in this range
    rows = np.zeros((max_n + 1, KMAX), dtype=np.uint16)
    for n in range(2, max_n + 1):
        recorded = []
        m = n
        while m >= n:
            if m % 2 == 1:
                m = 3 * m + 1
                recorded.append(v2(m))
            else:
                m //= 2
        row = rows[m].copy()
        for k in recorded:
            row[k] += 1
        rows[n] = row
    # The original pooled over ODD starting values only.
    pooled = rows[1::2].sum(axis=0, dtype=np.int64)
    return Counter({k: int(c) for k, c in enumerate(pooled) if c})


def report_v2(label, counts):
    total = sum(counts.values())
    print(f"\n  {label}  (total odd steps = {total:,})")
    print(f"  {'k':>3} {'observed':>10} {'geometric':>10} {'ratio':>8}")
    for k in range(1, 9):
        obs = counts.get(k, 0) / total
        exp = 2.0 ** (-k)
        ratio = obs / exp if exp else 0
        flag = "  <-- claimed excess" if k == 4 else ""
        print(f"  {k:>3} {obs:>10.5f} {exp:>10.5f} {ratio:>8.3f}{flag}")
    # max relative deviation across k=1..6
    devs = [abs(counts.get(k, 0) / total - 2.0 ** -k) / 2.0 ** -k for k in range(1, 7)]
    print(f"  max relative deviation (k=1..6): {max(devs) * 100:.1f}%")


def claim_a(full=False):
    print("=" * 70)
    print("CLAIM A:  v_2 = 4 excess after 3n+1")
    print("=" * 70)

    # 1. Confirm the theoretical null on uniform odds (the honest baseline).
    report_v2("NULL: uniform random odd n  (theory says exactly geometric)",
              v2_null_uniform_odds(2_000_000))

    # 2. Reproduce the original measurement on trajectory-visited odds.
    report_v2("ORIGINAL: trajectory-visited odds, n=1..100000  (biased subset)",
              v2_trajectory_visited(100_000))

    # 3. Scale check: does the bump decay as the range grows?
    # Four-point series cited in the paper: 1.721 / 1.526 / 1.424 / 1.353.
    print("\n  SCALE SERIES: k=4 ratio over trajectory-visited odds")
    for N in (1_000, 10_000, 100_000) + ((1_000_000,) if full else ()):
        c = v2_trajectory_visited(N)
        tot = sum(c.values())
        print(f"    N = {N:>9,}:  k=4 ratio = {(c.get(4, 0) / tot) / 2.0 ** -4:.3f}")
    if full:
        report_v2("SCALE CHECK: trajectory-visited odds, n=1..1000000",
                  v2_trajectory_visited(1_000_000))

    print("""
  INTERPRETATION:
  - If the uniform-odd null is flat (ratio ~ 1.000 everywhere) and the
    trajectory-visited version shows the k=4 bump, the "excess" is a property
    of WHICH numbers trajectories visit, not of the 3n+1 map producing
    unusually-divisible-by-16 outputs. The paper's mechanism claim ("3n+1
    creates acceleration lanes") would be unsupported as stated.
  - A bump that decays toward 1 as the range grows is a finite-range artifact,
    not a structural constant.""")


# ============================================================================= #
# CLAIM B: Benford "violation"
# ============================================================================= #
def benford_expected(d):
    return math.log10(1 + 1 / d)


def collatz_leading_digits(max_n):
    """Pooled leading-digit counts over all trajectory values of n = 1..max_n,
    via the same memoized recurrence as 02_benford_analysis.py."""
    import numpy as np
    counts = np.zeros((max_n + 1, 9), dtype=np.uint16)
    counts[1, 0] = 1
    for n in range(2, max_n + 1):
        recorded = []
        m = n
        while m >= n:
            recorded.append(leading_digit(m))
            m = m // 2 if m % 2 == 0 else 3 * m + 1
        row = counts[m].copy()
        for d in recorded:
            row[d - 1] += 1
        counts[n] = row
    pooled = counts[1:max_n + 1].sum(axis=0, dtype=np.int64)
    return Counter({d + 1: int(c) for d, c in enumerate(pooled)})


def chi_squared(counts):
    total = sum(counts.values())
    chi = 0.0
    for d in range(1, 10):
        exp = benford_expected(d) * total
        chi += (counts.get(d, 0) - exp) ** 2 / exp
    return chi, total


def total_variation_distance(counts):
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(d, 0) / total - benford_expected(d)) for d in range(1, 10))


def geometric_walk_leading_digits(n_walks, steps_each):
    """
    Finite-range NULL for Benford. A pure multiplicative random walk in log space
    converges to Benford only in the limit. Over the SAME finite ranges Collatz
    explores, does a comparable random walk ALSO miss Benford (esp. at digit 3)?
    If yes, the "violation" is a finite-range artifact, not Collatz-specific.
    """
    counts = Counter()
    for _ in range(n_walks):
        x = random.uniform(1, 10)
        for _ in range(steps_each):
            # multiply by 3/2 or by 1/2 with equal prob: same drift family as Collatz
            x *= 1.5 if random.random() < 0.5 else 0.5
            if x < 1:
                x *= 10 ** (math.ceil(-math.log10(x)))  # keep in a finite mantissa band
            counts[leading_digit(int(x) if x >= 1 else 1)] += 1
    return counts


def report_per_digit(counts, label):
    total = sum(counts.values())
    print(f"\n  {label} ({total:,} values):")
    print(f"  {'d':>3} {'observed':>10} {'benford':>10} {'dev %':>8}")
    for d in range(1, 10):
        obs = counts.get(d, 0) / total
        exp = benford_expected(d)
        print(f"  {d:>3} {obs:>10.5f} {exp:>10.5f} {(obs - exp) / exp * 100:>+8.1f}")


def claim_b(full=False):
    print("\n" + "=" * 70)
    print("CLAIM B:  Benford 'violation', chi^2 inflation, digit-3 dip")
    print("=" * 70)

    # 1. Show chi^2 scales with N (significance is an artifact of sample size).
    scales = (1_000, 10_000, 100_000) + ((1_000_000,) if full else ())
    print("\n  chi^2 grows ~linearly with N for the SAME deviation:")
    print(f"  {'N':>9} {'total values':>14} {'chi^2':>12} {'TVD':>8}")
    largest = None
    for N in scales:
        c = collatz_leading_digits(N)
        chi, total = chi_squared(c)
        tvd = total_variation_distance(c)
        print(f"  {N:>9,} {total:>14,} {chi:>12.1f} {tvd:>8.4f}")
        largest = c
    print("  (TVD = total variation distance from Benford; this is the real effect"
          " size and it barely moves while chi^2 explodes.)")

    # 2. Effect size + per-digit deviation at the largest N computed.
    report_per_digit(largest, f"Per-digit deviation, n=1..{scales[-1]:,}")

    # 3. Finite-range null: does a comparable random walk also miss Benford?
    g = geometric_walk_leading_digits(20_000, 200)
    report_per_digit(g, "Finite-range geometric-walk NULL")
    print(f"  geometric-walk TVD from Benford: {total_variation_distance(g):.4f}")
    print("""
  INTERPRETATION:
  - chi^2 = 61,154 is a sample-size statement, not an effect-size one.
  - The honest claim is the TVD and the per-digit deviations.
  - If the random walk ALSO shows a digit-3-ish dip at finite range, the
    Collatz 'violation' is largely a finite-range phenomenon, not evidence
    of special 3n+1 structure.""")


# ============================================================================= #
# CLAIM C: prime vs composite stopping-time gap (parity + size confounds)
# ============================================================================= #
def claim_c():
    print("\n" + "=" * 70)
    print("CLAIM C:  primes have ~4% longer trajectories than composites")
    print("=" * 70)
    import numpy as np

    st, sieve, n_max = load_stopping_times()
    n = np.arange(n_max + 1)
    is_odd = (n % 2) == 1
    valid = n > 1                       # 1 is neither prime nor composite
    prime = sieve & valid
    comp = ~sieve & valid

    def m(mask):
        return st[mask].mean(), int(mask.sum())

    p_mean, p_cnt = m(prime)
    c_mean, c_cnt = m(comp)
    print(f"\n  ORIGINAL COMPARISON (what the paper reports):")
    print(f"    primes      ({p_cnt:>7,}): mean ST = {p_mean:.2f}")
    print(f"    composites  ({c_cnt:>7,}): mean ST = {c_mean:.2f}")
    print(f"    gap: {(p_mean / c_mean - 1) * 100:+.1f}%")

    print(f"\n  PARITY CONFOUND: all primes but 2 are odd; composites are mostly even.")
    op_mean, op_cnt = m(prime & is_odd)
    oc_mean, oc_cnt = m(comp & is_odd)
    ec_mean, ec_cnt = m(comp & ~is_odd)
    print(f"    odd primes      ({op_cnt:>7,}): mean ST = {op_mean:.2f}")
    print(f"    odd composites  ({oc_cnt:>7,}): mean ST = {oc_mean:.2f}")
    print(f"    even composites ({ec_cnt:>7,}): mean ST = {ec_mean:.2f}")
    gap = op_mean / oc_mean - 1
    print(f"    parity-controlled gap (odd primes vs odd composites): {gap * 100:+.2f}%")

    # Effect size with a 95% CI (Welch), reported as a CI rather than a p-value.
    a = st[prime & is_odd].astype(np.float64)
    b = st[comp & is_odd].astype(np.float64)
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"    difference: {diff:+.2f} steps, 95% CI [{diff - 1.96 * se:+.2f}, {diff + 1.96 * se:+.2f}]")

    # Size control: prime density falls like 1/ln(n) while mean ST grows ~ln(n),
    # so compare within decades, odd numbers only.
    print(f"\n  SIZE CONTROL (odd numbers only, by decade):")
    print(f"    {'range':>20} {'odd primes':>12} {'odd comps':>12} {'gap %':>8}")
    for k in range(1, 6):
        lo, hi = 10 ** k, 10 ** (k + 1)
        band = (n >= lo) & (n < hi)
        ap, cp = st[prime & is_odd & band], st[comp & is_odd & band]
        if len(ap) == 0 or len(cp) == 0:
            continue
        print(f"    [{lo:>8,}, {hi:>8,}) {ap.mean():>12.2f} {cp.mean():>12.2f} "
              f"{(ap.mean() / cp.mean() - 1) * 100:>+8.2f}")

    print("""
  INTERPRETATION:
  - If the parity-controlled gap is far below the headline 4% and the
    per-decade gaps hover near 0, the prime/composite "effect" is mostly the
    odd/even composition of the two groups (odd starts climb via 3n+1 first),
    not primality.""")


# ============================================================================= #
# CLAIM D: Sophie Germain ratio ST(2p+1)/ST(p) ~ 4/3
# ============================================================================= #
def claim_d():
    print("\n" + "=" * 70)
    print("CLAIM D:  Sophie Germain ratio ST(2p+1)/ST(p) ~ 4/3")
    print("=" * 70)
    import numpy as np

    st, sieve, n_max = load_stopping_times()
    cap_pair = (n_max - 1) // 2          # need 2p+1 <= n_max
    p_all = np.nonzero(sieve)[0]
    p_all = p_all[(p_all <= cap_pair) & (p_all >= 2)]
    sg_mask = sieve[2 * p_all + 1]
    sg, non_sg = p_all[sg_mask], p_all[~sg_mask]
    odd_m = np.arange(3, cap_pair + 1, 2)
    odd_comp = odd_m[~sieve[odd_m]]

    def stats(ms, cap, label):
        ms = ms[(ms <= cap) & (st[ms] > 0)]
        r = st[2 * ms + 1] / st[ms]
        rom = st[2 * ms + 1].sum() / st[ms].sum()
        print(f"    {label:<38} n={len(ms):>7,}  mean(r)={r.mean():>6.3f}  "
              f"median(r)={np.median(r):>6.3f}  ratio-of-means={rom:>6.3f}")
        return r

    print(f"\n  The original statistic: MEAN OF RATIOS over SG primes p <= 50,000.")
    print(f"  Small p have tiny denominators (ST(2)=1, ST(3)=7, ST(5)=5), which")
    print(f"  inflates a mean of ratios. Smallest SG primes and their ratios:")
    for p in sg[:6]:
        print(f"    p={p:>5}  ST(p)={st[p]:>4}  ST(2p+1)={st[2 * p + 1]:>4}  "
              f"ratio={st[2 * p + 1] / st[p]:.3f}")

    print(f"\n  ESTIMATOR + CONTROL TABLE (map m -> 2m+1, same statistic everywhere):")
    print(f"  -- replication scale, cap = 50,000 (the original computation):")
    stats(sg, 50_000, "Sophie Germain primes")
    stats(non_sg, 50_000, "non-SG primes (2p+1 composite)")
    stats(odd_comp, 50_000, "odd composites (no primality at all)")
    print(f"  -- full scale, cap = {cap_pair:,}:")
    stats(sg, cap_pair, "Sophie Germain primes")
    stats(non_sg, cap_pair, "non-SG primes (2p+1 composite)")
    stats(odd_comp, cap_pair, "odd composites (no primality at all)")
    print(f"  -- small-denominator sensitivity (SG primes, p >= 1,000, cap 50,000):")
    stats(sg[sg >= 1000], 50_000, "Sophie Germain primes, p >= 1000")

    print("""
  INTERPRETATION:
  - If non-SG primes and odd composites give the SAME mean ratio as SG primes,
    the statistic measures the map m -> 2m+1 (and the mean-of-ratios estimator),
    not Sophie Germain structure.
  - If trimming small p (or switching to ratio-of-means / median) moves the
    value well away from 1.33, the "~4/3, close to log4/log3" reading is
    numerology on an estimator artifact. (Note also 4/3 = 1.333 and
    log4/log3 = 1.262 are different numbers five percent apart.)""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="also run claims A and B at N = 10^6 (the paper's scale check)")
    args = ap.parse_args()
    claim_a(full=args.full)
    claim_b(full=args.full)
    if os.path.exists(CSV_PATH):
        claim_c()
        claim_d()
    else:
        print(f"\n[skip] claims C and D need {CSV_PATH} (run 01_generate_data.py first)")
    print("\nDone.")
