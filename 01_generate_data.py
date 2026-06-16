"""
Phase 1: Generate Collatz sequence data for n = 1 to N.
Computes stopping time, total stopping time, max altitude, and trajectory length.
Saves results to CSV and binary formats for fast analysis.
"""
import time
import csv
import json
import os

def collatz_stats(n):
    """Compute Collatz trajectory statistics for a starting number n."""
    original = n
    steps = 0
    max_val = n
    stopping_time = None  # steps to first reach a value < n

    while n != 1:
        if n < original and stopping_time is None:
            stopping_time = steps
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
        if n > max_val:
            max_val = n

    if stopping_time is None:
        stopping_time = steps

    return {
        'n': original,
        'total_stopping_time': steps,
        'stopping_time': stopping_time,
        'max_altitude': max_val,
        'altitude_ratio': max_val / original if original > 0 else 0,
    }

def is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def main():
    N = 1_000_000  # 1 million numbers

    print(f"Generating Collatz data for n = 1 to {N:,}")
    start = time.time()

    # Results storage
    results = []

    # Track records
    max_stopping_time = 0
    max_stopping_n = 1
    max_altitude_val = 0
    max_altitude_n = 1
    max_total_stopping = 0
    max_total_stopping_n = 1

    # Stopping time distribution
    stopping_dist = {}
    total_stopping_dist = {}

    # Stats by residue class
    residue_stats = {mod: {} for mod in [2, 3, 6, 8, 12, 16, 24]}

    # Prime vs composite
    prime_stopping_times = []
    composite_stopping_times = []

    for n in range(1, N + 1):
        stats = collatz_stats(n)

        st = stats['stopping_time']
        tst = stats['total_stopping_time']
        ma = stats['max_altitude']

        # Track records
        if st > max_stopping_time:
            max_stopping_time = st
            max_stopping_n = n
        if ma > max_altitude_val:
            max_altitude_val = ma
            max_altitude_n = n
        if tst > max_total_stopping:
            max_total_stopping = tst
            max_total_stopping_n = n

        # Distribution
        stopping_dist[st] = stopping_dist.get(st, 0) + 1
        total_stopping_dist[tst] = total_stopping_dist.get(tst, 0) + 1

        # Residue class stats
        for mod in residue_stats:
            r = n % mod
            if r not in residue_stats[mod]:
                residue_stats[mod][r] = {'count': 0, 'total_st': 0, 'total_tst': 0, 'total_ma': 0}
            residue_stats[mod][r]['count'] += 1
            residue_stats[mod][r]['total_st'] += st
            residue_stats[mod][r]['total_tst'] += tst
            residue_stats[mod][r]['total_ma'] += ma

        # Prime vs composite
        if n > 2:
            if is_prime(n):
                prime_stopping_times.append(tst)
            else:
                composite_stopping_times.append(tst)

        # Store compact results (every number)
        results.append((n, st, tst, ma))

        if n % 100000 == 0:
            elapsed = time.time() - start
            print(f"  {n:>10,} / {N:,} ({n/N*100:.0f}%) — {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Save compact CSV
    csv_path = os.path.join(out_dir, 'collatz_data.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'stopping_time', 'total_stopping_time', 'max_altitude'])
        for row in results:
            w.writerow(row)
    print(f"Saved {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.1f} MB)")

    # Save summary statistics
    summary = {
        'N': N,
        'computation_time_seconds': round(elapsed, 2),
        'records': {
            'max_stopping_time': {'value': max_stopping_time, 'n': max_stopping_n},
            'max_total_stopping_time': {'value': max_total_stopping, 'n': max_total_stopping_n},
            'max_altitude': {'value': max_altitude_val, 'n': max_altitude_n},
        },
        'stopping_time_distribution': {str(k): v for k, v in sorted(stopping_dist.items())},
        'total_stopping_time_distribution': {str(k): v for k, v in sorted(total_stopping_dist.items())[:100]},
        'residue_class_averages': {},
        'prime_vs_composite': {
            'prime_count': len(prime_stopping_times),
            'prime_avg_total_stopping': round(sum(prime_stopping_times) / len(prime_stopping_times), 4) if prime_stopping_times else 0,
            'composite_count': len(composite_stopping_times),
            'composite_avg_total_stopping': round(sum(composite_stopping_times) / len(composite_stopping_times), 4) if composite_stopping_times else 0,
        }
    }

    # Compute residue class averages
    for mod in residue_stats:
        summary['residue_class_averages'][str(mod)] = {}
        for r in sorted(residue_stats[mod].keys()):
            s = residue_stats[mod][r]
            summary['residue_class_averages'][str(mod)][str(r)] = {
                'count': s['count'],
                'avg_stopping_time': round(s['total_st'] / s['count'], 4),
                'avg_total_stopping_time': round(s['total_tst'] / s['count'], 4),
                'avg_max_altitude': round(s['total_ma'] / s['count'], 2),
            }

    summary_path = os.path.join(out_dir, 'collatz_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")

    # Print key findings
    print(f"\n{'='*60}")
    print(f"KEY FINDINGS (n = 1 to {N:,})")
    print(f"{'='*60}")
    print(f"Max stopping time: {max_stopping_time} (n = {max_stopping_n})")
    print(f"Max total stopping time: {max_total_stopping} (n = {max_total_stopping_n})")
    print(f"Max altitude: {max_altitude_val:,} (n = {max_altitude_n})")
    print(f"Prime avg total stopping: {summary['prime_vs_composite']['prime_avg_total_stopping']}")
    print(f"Composite avg total stopping: {summary['prime_vs_composite']['composite_avg_total_stopping']}")

    # Residue class mod 6 analysis
    print(f"\nResidue classes mod 6:")
    for r in sorted(residue_stats[6].keys()):
        s = residue_stats[6][r]
        avg = s['total_tst'] / s['count']
        print(f"  n ≡ {r} (mod 6): avg total stopping time = {avg:.2f}")

if __name__ == '__main__':
    main()
