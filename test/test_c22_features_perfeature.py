"""
Per-feature diagnostic: compare CPU features (via pycatch22) with GPU profile
on a carefully crafted synthetic series where we can measure the diff.

Usage (on cerebro):
    PYTHONPATH=build python3 test/test_c22_features_perfeature.py
"""
import sys
import numpy as np

try:
    import pyscamp
except ImportError:
    print("pyscamp not found — run from build directory or set PYTHONPATH")
    sys.exit(1)

try:
    import pycatch22
    HAS_PYCATCH22 = True
except ImportError:
    HAS_PYCATCH22 = False
    print("pycatch22 not available — will only check profile-level GPU vs CPU diff")

def zscore(x):
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return x - m
    return (x - m) / s

def cpu_dot(fa, fb):
    return float(np.dot(fa, fb))

def brute_force_profile(features, excl):
    N = len(features)
    profile = np.full(N, -np.inf)
    index   = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if abs(i - j) <= excl:
                continue
            d = cpu_dot(features[i], features[j])
            if d > profile[i]:
                profile[i] = d
                index[i] = j
    return profile, index

np.random.seed(42)

W = 50
N_sub = 200
ts = list(np.cumsum(np.random.randn(N_sub + W - 1)))
excl = W // 4

print(f"W={W}, N={N_sub}, excl={excl}")
print()

# ── CPU profile ──────────────────────────────────────────────────────────────
print("Computing CPU profile... ", end="", flush=True)
p_cpu, idx_cpu = pyscamp.selfjoin_c22(ts, W, threads=8, use_gpu=False)
p_cpu = np.array(p_cpu)
print("done")

# ── GPU profile ──────────────────────────────────────────────────────────────
print("Computing GPU profile... ", end="", flush=True)
p_gpu, idx_gpu = pyscamp.selfjoin_c22(ts, W, threads=8, use_gpu=True)
p_gpu = np.array(p_gpu)
print("done")

diff = np.abs(p_gpu - p_cpu)
print(f"\nProfile max_diff = {diff.max():.4g},  mean_diff = {diff.mean():.4g}")
print(f"# rows with diff > 0.001 = {(diff > 1e-3).sum()} / {N_sub}")

worst5 = np.argsort(diff)[-5:][::-1]
print(f"\nTop-5 worst rows (index, diff):")
for r in worst5:
    print(f"  row={r:4d}  diff={diff[r]:.4g}  cpu_profile={p_cpu[r]:.6g}  gpu_profile={p_gpu[r]:.6g}")
    print(f"           cpu_idx={idx_cpu[r]}   gpu_idx={idx_gpu[r]}")

# ── Per-feature breakdown (needs pycatch22) ───────────────────────────────────
if HAS_PYCATCH22:
    print("\n\nPer-feature analysis...")
    ts_arr = np.array(ts)
    subseqs = [zscore(ts_arr[i:i+W]) for i in range(N_sub)]

    print("Computing pycatch22 features for all subseqs... ", end="", flush=True)
    feat_names = pycatch22.catch22_all(list(subseqs[0]))["names"]
    features_cpu = []
    for sub in subseqs:
        r = pycatch22.catch22_all(list(sub))
        features_cpu.append(np.array(r["values"], dtype=float))
    features_cpu = np.array(features_cpu)  # N × 22
    print("done")

    # Check NaN/Inf
    bad = np.any(~np.isfinite(features_cpu), axis=0)
    for fi, name in enumerate(feat_names):
        if bad[fi]:
            print(f"  Feature {fi:2d} ({name}) has NaN/Inf in {(~np.isfinite(features_cpu[:,fi])).sum()} subseqs")

    # Brute-force profile from CPU features
    print("Brute-force CPU profile from pycatch22 features... ", end="", flush=True)
    p_ref, idx_ref = brute_force_profile(features_cpu, excl)
    print("done")
    
    diff_ref = np.abs(p_cpu - p_ref)
    print(f"\nCPU-profile vs pycatch22-brute-force: max_diff={diff_ref.max():.4g}")

    # For the top-5 worst GPU rows, compute per-feature contribution to error
    print("\nPer-feature contribution for worst rows:")
    for r in worst5[:3]:
        j_cpu = int(idx_cpu[r])
        j_gpu = int(idx_gpu[r])
        fi_r = features_cpu[r]   # features of row r
        fi_jcpu = features_cpu[j_cpu]
        fi_jgpu = features_cpu[j_gpu] if j_gpu < N_sub else fi_r

        # per-feature dot contribution
        contrib_cpu = fi_r * fi_jcpu
        contrib_gpu = fi_r * fi_jgpu if j_cpu != j_gpu else contrib_cpu

        print(f"\n  Row {r}: diff={diff[r]:.4g}  j_cpu={j_cpu}  j_gpu={j_gpu}")
        if j_cpu != j_gpu:
            diff_by_feat = np.abs(contrib_cpu - contrib_gpu)
            top_feats = np.argsort(diff_by_feat)[-5:][::-1]
            for fi in top_feats:
                print(f"    feat[{fi:2d}] {feat_names[fi]:<45s}: "
                      f"row={fi_r[fi]:+.4f}  j_cpu={fi_jcpu[fi]:+.4f}  j_gpu={fi_jgpu[fi]:+.4f}"
                      f"  contrib_diff={diff_by_feat[fi]:.4g}")
        else:
            print(f"    Same neighbor j={j_cpu}, diff is in profile value itself")
            # Both use same neighbor, diff = p_gpu[r] - p_cpu[r]
            # This means GPU computed wrong feature values for (r, j_cpu)

    # Final summary: which feature has highest RMSE across all rows
    print("\n\nFeature magnitude stats across all subsequences:")
    print(f"{'idx':>3} {'name':<45} {'mean':>8} {'std':>8} {'max_abs':>8}")
    for fi, name in enumerate(feat_names):
        vals = features_cpu[:, fi]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        print(f"  {fi:2d} {name:<45} {vals.mean():+8.3f} {vals.std():8.3f} {np.max(np.abs(vals)):8.3f}")
else:
    print("\npycatch22 not available — install with: pip3 install pycatch22 --user")
    print("Without it, we can only see the profile-level diff above.")
