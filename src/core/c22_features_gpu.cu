// c22_features_gpu.cu
//
// GPU implementation of all 22 catch22 features.
//
// Design:
//   - One CUDA thread per subsequence.
//   - All temporary storage is stack-allocated (local memory / L1 cache).
//   - Maximum supported window size: C22_GPU_MAX_W (256).  Windows larger than
//     this fall back to the CPU path automatically (handled in
//     c22_profile.cpp).
//   - Autocorrelation uses direct O(W²) loops instead of FFT — simpler device
//     code and competitive for W ≤ 256.
//   - Spectral features (SP_Summaries) use an in-register iterative FFT.
//   - PD_PeriodicityWang uses the full spline fit (exact port of
//     pycatch22/src/C/splinefit.c) and cov_mean autocorrelation matching the
//     CPU implementation.  All 22 features should match the CPU exactly.
//
// Stack per thread at W=256: ~58 KB (dominated by DN_OutlierInclude working
// arrays and the inlined device function locals).  We set
//   cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024)   ← 128 KB
// before launching to give ample headroom above the ~58 KB actual usage.
// This is done in c22_compute_features_gpu_launch() and in
// c22_profile_selfjoin_gpu_launch_from_ts() / _abjoin_.

#ifdef _HAS_CUDA_

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================
#define C22_GPU_MAX_W 256    // max window size for GPU feature extraction
#define C22_GPU_SP_NFFT 256  // FFT size for SP_Summaries (nextpow2(MAX_W))
#define C22_GPU_MAX_NTHRESH \
  300                          // max outlier threshold levels (z-scored max≈3)
#define C22_GPU_BLOCK_SIZE 64  // threads per block
#define C22_GPU_NUM_FEAT 22

// ============================================================================
// ── Math utilities (device) ──────────────────────────────────────────────────
// ============================================================================

__device__ static double gpu_mean(const double* y, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += y[i];
  return s / n;
}

__device__ static double gpu_var_given_mean(const double* y, int n, double m) {
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    double d = y[i] - m;
    s += d * d;
  }
  return s / (n - 1);
}

__device__ static double gpu_stddev(const double* y, int n) {
  double m = gpu_mean(y, n);
  return sqrt(gpu_var_given_mean(y, n, m));
}

__device__ static double gpu_min(const double* y, int n) {
  double m = y[0];
  for (int i = 1; i < n; i++)
    if (y[i] < m) m = y[i];
  return m;
}

__device__ static double gpu_max(const double* y, int n) {
  double m = y[0];
  for (int i = 1; i < n; i++)
    if (y[i] > m) m = y[i];
  return m;
}

// z-score normalize src → dst
__device__ static void gpu_zscore(const double* src, int n, double* dst) {
  double m = gpu_mean(src, n);
  double sd = sqrt(gpu_var_given_mean(src, n, m));
  if (sd < 1e-12) sd = 1.0;
  for (int i = 0; i < n; i++) dst[i] = (src[i] - m) / sd;
}

// In-place insertion sort (ascending)
__device__ static void gpu_sort(double* y, int n) {
  for (int i = 1; i < n; i++) {
    double key = y[i];
    int j = i - 1;
    while (j >= 0 && y[j] > key) {
      y[j + 1] = y[j];
      j--;
    }
    y[j + 1] = key;
  }
}

// Median of a pre-sorted array copy (sorts in-place)
__device__ static double gpu_median(const double* y, int n,
                                    double* tmp) {  // tmp must be n doubles
  for (int i = 0; i < n; i++) tmp[i] = y[i];
  gpu_sort(tmp, n);
  if (n & 1) return tmp[n / 2];
  return (tmp[n / 2 - 1] + tmp[n / 2]) * 0.5;
}

// quantile q ∈ [0,1] — sorts a copy into tmp
__device__ static double gpu_quantile(const double* y, int n, double q,
                                      double* tmp) {
  for (int i = 0; i < n; i++) tmp[i] = y[i];
  gpu_sort(tmp, n);
  double qi = q * 0.5 / n;
  if (q < qi) return tmp[0];
  if (q > 1.0 - qi) return tmp[n - 1];
  double qidx = n * q - 0.5;
  int left = (int)floor(qidx);
  int right = (int)ceil(qidx);
  if (left == right) return tmp[left];
  return tmp[left] + (qidx - left) * (tmp[right] - tmp[left]);
}

// Simple OLS: y = m*x + b (x = 0,1,...,n-1)
__device__ static void gpu_linreg(const double* x, const double* y, int n,
                                  double* m_out, double* b_out) {
  double sx = 0, sx2 = 0, sxy = 0, sy = 0;
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sx2 += x[i] * x[i];
    sxy += x[i] * y[i];
    sy += y[i];
  }
  double denom = n * sx2 - sx * sx;
  if (fabs(denom) < 1e-30) {
    *m_out = 0;
    *b_out = 0;
    return;
  }
  *m_out = (n * sxy - sx * sy) / denom;
  *b_out = (sy * sx2 - sx * sxy) / denom;
}

// Raw autocov sum at lag: sum(y[i]*y[i+lag]) — NOT divided by (n-lag).
// Dividing by the lag-0 value (sum of squares) reproduces pycatch22's
// co_autocorrs normalisation: ac[k] = sum(y*y[+k]) / sum(y²).
__device__ static double gpu_autocov_lag(const double* y, int n, int lag) {
  if (lag >= n) return 0.0;
  double s = 0.0;
  for (int i = 0; i < n - lag; i++) s += y[i] * y[i + lag];
  return s;  // raw sum — caller divides by lag-0 for normalisation
}

// Autocovariance at lag: matches pycatch22's autocov_lag = cov_mean(x, &x[lag], size-lag).
// cov_mean(x, y, n) = sum(x[i]*y[i]) / n  — NO mean subtraction, divides by n.
// Used only by PD_PeriodicityWang.
__device__ static double gpu_autocov_cov_mean(const double* y, int n, int lag) {
  if (lag >= n) return 0.0;
  int sz = n - lag;
  double s = 0.0;
  for (int i = 0; i < sz; i++) s += y[i] * y[i + lag];
  return s / sz;
}

// Pearson autocorrelation at lag: corr(y[0:n-lag], y[lag:n]).
// Matches pycatch22's autocorr_lag() in stats.c, which calls
// corr(x, &x[lag], size-lag) — a full Pearson with per-slice mean+std.
// Used ONLY by IN_AutoMutualInfoStats_40_gaussian_fmmi.
__device__ static double gpu_autocorr_pearson_lag(const double* y, int n,
                                                  int lag) {
  int sz = n - lag;
  if (sz <= 0) return 0.0;
  const double* xa = y;        // y[0..sz-1]
  const double* xb = y + lag;  // y[lag..n-1]
  double mx = 0.0, my = 0.0;
  for (int i = 0; i < sz; i++) {
    mx += xa[i];
    my += xb[i];
  }
  mx /= sz;
  my /= sz;
  double nom = 0.0, dX = 0.0, dY = 0.0;
  for (int i = 0; i < sz; i++) {
    double dx = xa[i] - mx, dy = xb[i] - my;
    nom += dx * dy;
    dX += dx * dx;
    dY += dy * dy;
  }
  double denom = sqrt(dX * dY);
  return (denom < 1e-30) ? 0.0 : nom / denom;
}

// Compute autocorrelations for lags 0..maxlag into out[] (size maxlag+1)
// Formula: out[k] = sum(y[i]*y[i+k]) / sum(y[i]²)  — matches pycatch22
__device__ static void gpu_co_autocorrs(const double* y, int n, double* out,
                                        int maxlag) {
  double var = gpu_autocov_lag(y, n, 0);  // = sum(y²)
  double inv_var = (fabs(var) < 1e-30) ? 0.0 : 1.0 / var;
  for (int lag = 0; lag <= maxlag; lag++)
    out[lag] = gpu_autocov_lag(y, n, lag) * inv_var;
}

// First zero-crossing of autocorrelation (≥ 1)
__device__ static int gpu_co_firstzero(const double* y, int n, const double* ac,
                                       int maxlag) {
  for (int i = 0; i < maxlag - 1; i++)
    if (ac[i + 1] <= 0.0) return i + 1;
  return maxlag;
}

// ============================================================================
// ── Iterative FFT (power-of-2 size, for SP_Summaries only) ──────────────────
// ============================================================================
// In-place Cooley-Tukey, operates on separate real/imag arrays.

__device__ static int gpu_nextpow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

__device__ static void gpu_fft(double* re, double* im, int n) {
  // bit-reversal
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double t;
      t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  // butterfly
  const double PI = 3.14159265358979323846;
  for (int len = 2; len <= n; len <<= 1) {
    double ang = -2.0 * PI / len;
    double wre = cos(ang), wim = sin(ang);
    for (int i = 0; i < n; i += len) {
      double curre = 1.0, curim = 0.0;
      for (int k = 0; k < len / 2; k++) {
        double ur = re[i + k], ui = im[i + k];
        double vr = curre * re[i + k + len / 2] - curim * im[i + k + len / 2];
        double vi = curre * im[i + k + len / 2] + curim * re[i + k + len / 2];
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        double nr = curre * wre - curim * wim;
        double ni = curre * wim + curim * wre;
        curre = nr;
        curim = ni;
      }
    }
  }
}

// ============================================================================
// ── Coarse-graining (for SB_TransitionMatrix, SB_MotifThree) ────────────────
// ============================================================================
// Assigns each element a quantile label 1..num_groups using a sorted copy tmp.
__device__ static void gpu_coarsegrain_quantile(const double* y, int n,
                                                int num_groups, int* labels,
                                                double* tmp) {
  // compute quantile thresholds: 0, 1/g, 2/g, ..., 1
  double th[4];  // num_groups+1 ≤ 4
  for (int k = 0; k <= num_groups; k++) {
    double q = (double)k / num_groups;
    th[k] = gpu_quantile(y, n, q, tmp);
  }
  th[0] -= 1.0;  // open lower bound (match pycatch22 behaviour)
  for (int i = 0; i < n; i++) {
    labels[i] = 0;
    for (int k = 0; k < num_groups; k++) {
      if (y[i] > th[k] && y[i] <= th[k + 1]) {
        labels[i] = k + 1;
        break;
      }
    }
  }
}

// ============================================================================
// ── Feature 1: DN_HistogramMode_5 ───────────────────────────────────────────
// ============================================================================
__device__ static double gpu_DN_HistogramMode(const double* z, int W,
                                              int nBins) {
  double mn = gpu_min(z, W), mx = gpu_max(z, W);
  if (mx - mn < 1e-30) return 0.0;
  double step = (mx - mn) / nBins;
  int counts[10] = {0};  // nBins ≤ 10
  double edges[11];
  for (int b = 0; b <= nBins; b++) edges[b] = mn + b * step;
  for (int i = 0; i < W; i++) {
    int b = (int)((z[i] - mn) / step);
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    counts[b]++;
  }
  double maxC = 0, out = 0;
  int numMaxs = 0;
  for (int b = 0; b < nBins; b++) {
    double centre = (edges[b] + edges[b + 1]) * 0.5;
    if (counts[b] > maxC) {
      maxC = counts[b];
      numMaxs = 1;
      out = centre;
    } else if (counts[b] == maxC) {
      numMaxs++;
      out += centre;
    }
  }
  return out / numMaxs;
}

// ============================================================================
// ── Feature 3: CO_f1ecac  (first 1/e crossing of AC) ────────────────────────
// ============================================================================
__device__ static double gpu_CO_f1ecac(const double* z, int W,
                                       const double* ac) {
  double thresh = 1.0 / exp(1.0);
  for (int i = 0; i < W - 2; i++) {
    if (ac[i + 1] < thresh) {
      double m = ac[i + 1] - ac[i];
      if (fabs(m) < 1e-30) return (double)i;
      return (double)i + (thresh - ac[i]) / m;
    }
  }
  return (double)W;
}

// ============================================================================
// ── Feature 4: CO_FirstMin_ac ────────────────────────────────────────────────
// ============================================================================
__device__ static int gpu_CO_FirstMin_ac(const double* ac, int W) {
  for (int i = 1; i < W - 1; i++)
    if (ac[i] < ac[i - 1] && ac[i] < ac[i + 1]) return i;
  return W;
}

// ============================================================================
// ── Feature 5: CO_HistogramAMI_even_2_5 ─────────────────────────────────────
// ============================================================================
__device__ static double gpu_CO_HistogramAMI_even_2_5(const double* z, int W) {
  const int tau = 2, nBins = 5;
  int sz = W - tau;
  double mn = gpu_min(z, W), mx = gpu_max(z, W);
  double binStep = (mx - mn + 0.2) / nBins;
  double edges[6];
  for (int b = 0; b <= nBins; b++) edges[b] = mn + b * binStep - 0.1;

  // assign bins for y1=z[0..sz-1], y2=z[tau..sz+tau-1]
  int bins1[C22_GPU_MAX_W], bins2[C22_GPU_MAX_W];
  for (int i = 0; i < sz; i++) {
    bins1[i] = 0;
    for (int b = 0; b < nBins + 1; b++)
      if (z[i] < edges[b]) {
        bins1[i] = b;
        break;
      }
    bins2[i] = 0;
    for (int b = 0; b < nBins + 1; b++)
      if (z[i + tau] < edges[b]) {
        bins2[i] = b;
        break;
      }
  }

  // joint histogram (nBins×nBins)
  double pij[5][5] = {{}};
  for (int i = 0; i < sz; i++) {
    int b1 = bins1[i] - 1, b2 = bins2[i] - 1;
    if (b1 >= 0 && b1 < nBins && b2 >= 0 && b2 < nBins) pij[b1][b2]++;
  }
  // normalise
  double tot = 0;
  for (int a = 0; a < nBins; a++)
    for (int b = 0; b < nBins; b++) tot += pij[a][b];
  if (tot < 1) return 0.0;
  for (int a = 0; a < nBins; a++)
    for (int b = 0; b < nBins; b++) pij[a][b] /= tot;

  // marginals
  double pi[5] = {}, pj[5] = {};
  for (int a = 0; a < nBins; a++)
    for (int b = 0; b < nBins; b++) {
      pi[a] += pij[a][b];
      pj[b] += pij[a][b];
    }

  double ami = 0.0;
  for (int a = 0; a < nBins; a++)
    for (int b = 0; b < nBins; b++)
      if (pij[a][b] > 0 && pi[a] > 0 && pj[b] > 0)
        ami += pij[a][b] * log(pij[a][b] / (pi[a] * pj[b]));
  return ami;
}

// ============================================================================
// ── Feature 6: CO_trev_1_num ─────────────────────────────────────────────────
// ============================================================================
__device__ static double gpu_CO_trev_1_num(const double* z, int W) {
  double s = 0.0;
  for (int i = 0; i < W - 1; i++) {
    double d = z[i + 1] - z[i];
    s += d * d * d;
  }
  return s / (W - 1);
}

// ============================================================================
// ── Feature 7: MD_hrv_classic_pnn40 ─────────────────────────────────────────
// ============================================================================
__device__ static double gpu_MD_hrv_pnn40(const double* z, int W) {
  int cnt = 0;
  for (int i = 0; i < W - 1; i++)
    if (fabs(z[i + 1] - z[i]) * 1000.0 > 40.0) cnt++;
  return (double)cnt / (W - 1);
}

// ============================================================================
// ── Feature 8: SB_BinaryStats_mean_longstretch1 ──────────────────────────────
// ============================================================================
__device__ static double gpu_SB_BinaryStats_mean_longstretch1(const double* z,
                                                              int W) {
  double m = gpu_mean(z, W);
  int maxStr = 0, last0 = 0;
  for (int i = 0; i < W - 1; i++) {
    int bin = (z[i] - m > 0) ? 1 : 0;
    if (bin == 0 || i == W - 2) {
      int stretch = i - last0;
      if (stretch > maxStr) maxStr = stretch;
      last0 = i;
    }
  }
  return (double)maxStr;
}

// ============================================================================
// ── Feature 9: SB_TransitionMatrix_3ac_sumdiagcov ────────────────────────────
// ============================================================================
__device__ static double gpu_SB_TransitionMatrix_3ac_sumdiagcov(
    const double* z, int W, const double* ac, double* tmp_sort) {
  // tau = first zero-crossing of AC
  int tau = 0;
  for (int i = 0; i < W; i++)
    if (ac[i + 1] <= 0.0) {
      tau = i + 1;
      break;
    }
  if (tau == 0) tau = W;

  // downsample
  int nDown = (W - 1) / tau + 1;
  double yDown[C22_GPU_MAX_W];
  for (int i = 0; i < nDown; i++) yDown[i] = z[i * tau];

  // coarse-grain (3 quantile groups)
  int yCG[C22_GPU_MAX_W];
  gpu_coarsegrain_quantile(yDown, nDown, 3, yCG, tmp_sort);

  // 3×3 transition matrix
  double T[3][3] = {{}};
  for (int j = 0; j < nDown - 1; j++) {
    int r = yCG[j] - 1, c = yCG[j + 1] - 1;
    if (r >= 0 && r < 3 && c >= 0 && c < 3) T[r][c]++;
  }
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) T[r][c] /= (nDown - 1);

  // columns
  double col0[3], col1[3], col2[3];
  for (int r = 0; r < 3; r++) {
    col0[r] = T[r][0];
    col1[r] = T[r][1];
    col2[r] = T[r][2];
  }

  // cov(col_i, col_i) = variance of each column (diagonal of 3×3 cov matrix)
  double sumdiagcov = 0.0;
  double* cols[3] = {col0, col1, col2};
  for (int k = 0; k < 3; k++) {
    double m = (cols[k][0] + cols[k][1] + cols[k][2]) / 3.0;
    double v = 0;
    for (int r = 0; r < 3; r++) {
      double d = cols[k][r] - m;
      v += d * d;
    }
    sumdiagcov += v / 2.0;  // n-1=2
  }
  return sumdiagcov;
}

// ============================================================================
// ── GPU Splinefit — exact port of pycatch22/src/C/splinefit.c ───────────────
// Fits a 2-piece cubic B-spline (pieces=2, deg=3, nSpline=4, piecesExt=8)
// with breakpoints at {0, floor(n/2)-1, n-1} to the data y[0..n-1].
// Writes the fitted spline values into yOut[0..n-1].
// Stack usage: ~29 KB (xsB/indexB/vB/Amat dominate).
// ============================================================================
__device__ __noinline__ static void gpu_splinefit(const double* y, int n,
                                                  double* yOut) {
  // breakpoints
  int br[3];
  br[0] = 0;
  br[1] = (int)floor((double)n / 2.0) - 1;
  br[2] = n - 1;

  int h0 = br[1] - br[0];  // spacing of first piece
  int h1 = br[2] - br[1];  // spacing of second piece

  // hCopy[4] = {h0, h1, h0, h1}
  int hCopy[4] = {h0, h1, h0, h1};

  // Extended breaks to the LEFT  (hl = hCopy reversed subset)
  // hl[0]=hCopy[3]=h1, hl[1]=hCopy[2]=h0, hl[2]=hCopy[1]=h1
  int hl[3], hlCS[3];
  hl[0] = hCopy[3];
  hl[1] = hCopy[2];
  hl[2] = hCopy[1];
  hlCS[0] = hl[0];
  hlCS[1] = hlCS[0] + hl[1];
  hlCS[2] = hlCS[1] + hl[2];
  int bl[3];
  for (int i = 0; i < 3; i++) bl[i] = br[0] - hlCS[i];

  // Extended breaks to the RIGHT
  // hr[0]=hCopy[0]=h0, hr[1]=hCopy[1]=h1, hr[2]=hCopy[2]=h0
  int hr[3], hrCS[3];
  hr[0] = hCopy[0];
  hr[1] = hCopy[1];
  hr[2] = hCopy[2];
  hrCS[0] = hr[0];
  hrCS[1] = hrCS[0] + hr[1];
  hrCS[2] = hrCS[1] + hr[2];
  int brr[3];
  for (int i = 0; i < 3; i++) brr[i] = br[2] + hrCS[i];

  // Full extended breakpoints (9 entries)
  int breaksExt[9];
  for (int i = 0; i < 3; i++) {
    breaksExt[i] = bl[2 - i];
    breaksExt[i + 3] = br[i];
    breaksExt[i + 6] = brr[i];
  }
  int hExt[8];
  for (int i = 0; i < 8; i++) hExt[i] = breaksExt[i + 1] - breaksExt[i];

  // Index matrix ii[4][8]: ii[r][c] = min(r+c, 7)
  int ii[4][8];
  for (int c = 0; c < 8; c++) {
    ii[0][c] = (c < 8) ? c : 7;
    ii[1][c] = (c + 1 < 8) ? c + 1 : 7;
    ii[2][c] = (c + 2 < 8) ? c + 2 : 7;
    ii[3][c] = (c + 3 < 8) ? c + 3 : 7;
  }

  // H[32]: H[l] = hExt[ii[l%4][l/4]]
  double H[32];
  for (int l = 0; l < 32; l++) H[l] = (double)hExt[ii[l % 4][l / 4]];

  // coefs[32][5] — B-spline polynomial coefficients (initialised to step fns)
  double coefs[32][5];
  for (int i = 0; i < 32; i++)
    for (int j = 0; j < 5; j++) coefs[i][j] = 0.0;
  for (int i = 0; i < 32; i += 4) coefs[i][0] = 1.0;

  double Q[4][8];

  // Recursive B-spline generation: build order-1..4 B-splines
  for (int k = 1; k < 4; k++) {
    // antiderivatives: scale coefs[*][0..k-1] by H/(k-j)
    for (int j = 0; j < k; j++)
      for (int l = 0; l < 32; l++) coefs[l][j] *= H[l] / (double)(k - j);

    // Q[row][col] = sum of coefs row for col
    for (int l = 0; l < 32; l++) {
      Q[l % 4][l / 4] = 0.0;
      for (int m = 0; m < 4; m++) Q[l % 4][l / 4] += coefs[l][m];
    }
    // cumsum Q along rows (column by column)
    for (int col = 0; col < 8; col++)
      for (int row = 1; row < 4; row++) Q[row][col] += Q[row - 1][col];

    // update coefs[*][k] from Q (Q[row-1] for row>0, 0 for row==0)
    for (int l = 0; l < 32; l++) {
      if (l % 4 == 0)
        coefs[l][k] = 0.0;
      else
        coefs[l][k] = Q[(l % 4) - 1][l / 4];
    }

    // normalise by fmax = Q[3][col]: coefs[l][0..k] /= Q[3][l/4]
    for (int j = 0; j <= k; j++)
      for (int l = 0; l < 32; l++) {
        double fmax = Q[3][l / 4];
        if (fabs(fmax) > 1e-30) coefs[l][j] /= fmax;
      }

    // diff to adjacent antiderivatives: coefs[l] -= coefs[l+3] for l=0..28
    for (int l = 0; l < 29; l++)
      for (int j = 0; j <= k; j++) coefs[l][j] -= coefs[l + 3][j];
    // zero out every 4th coef[k]
    for (int l = 0; l < 32; l += 4) coefs[l][k] = 0.0;
  }

  // Scale polynomial coefficients
  double scale[32];
  for (int i = 0; i < 32; i++) scale[i] = 1.0;
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < 32; i++) scale[i] /= H[i];
    for (int i = 0; i < 32; i++) coefs[i][3 - (k + 1)] *= scale[i];
  }

  // jj[4][2]: reduction index matrix
  int jj[4][2];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) jj[i][j] = (i == 0) ? 4 * (1 + j) : 3;
  // cumsum along rows
  for (int i = 1; i < 4; i++)
    for (int j = 0; j < 2; j++) jj[i][j] += jj[i - 1][j];

  // coefsOut[8][4]: extracted B-spline piece coefficients
  double coefsOut[8][4];
  for (int l = 0; l < 8; l++) {
    int jj_flat = jj[l % 4][l / 4] - 1;
    for (int j = 0; j < 4; j++) coefsOut[l][j] = coefs[jj_flat][j];
  }

  // Build basis matrix A (n × 5) using B-splines
  int xsB[C22_GPU_MAX_W * 4];
  int indexB[C22_GPU_MAX_W * 4];
  double vB[C22_GPU_MAX_W * 4];
  {
    int breakInd = 1;
    for (int i = 0; i < n; i++) {
      if (i >= br[breakInd] && breakInd < 2) breakInd++;
      for (int j = 0; j < 4; j++) {
        xsB[i * 4 + j] = i - br[breakInd - 1];
        indexB[i * 4 + j] = j + (breakInd - 1) * 4;
      }
    }
    for (int i = 0; i < n * 4; i++) vB[i] = coefsOut[indexB[i]][0];
    // Horner's method for basis evaluation
    for (int k = 1; k < 4; k++)
      for (int j = 0; j < n * 4; j++)
        vB[j] = vB[j] * (double)xsB[j] + coefsOut[indexB[j]][k];
  }

  // Fill A matrix (n rows, 5 columns)
  double Amat[C22_GPU_MAX_W * 5];
  for (int i = 0; i < 5 * n; i++) Amat[i] = 0.0;
  {
    int breakInd2 = 0;
    for (int i = 0; i < 4 * n; i++) {
      if (i / 4 >= br[1]) breakInd2 = 1;
      Amat[(i % 4) + breakInd2 + (i / 4) * 5] = vB[i];
    }
  }

  // Solve normal equations A^T*A * x = A^T*y  (5×5 system)
  double ATA[5][5], ATy[5];
  for (int r = 0; r < 5; r++) {
    for (int c = 0; c < 5; c++) {
      double s = 0.0;
      for (int k = 0; k < n; k++) s += Amat[k * 5 + r] * Amat[k * 5 + c];
      ATA[r][c] = s;
    }
    double s = 0.0;
    for (int k = 0; k < n; k++) s += Amat[k * 5 + r] * y[k];
    ATy[r] = s;
  }
  // Gaussian elimination on augmented [ATA | ATy]
  double aug[5][6];
  for (int r = 0; r < 5; r++) {
    for (int c = 0; c < 5; c++) aug[r][c] = ATA[r][c];
    aug[r][5] = ATy[r];
  }
  for (int col = 0; col < 5; col++)
    for (int row = col + 1; row < 5; row++) {
      if (fabs(aug[col][col]) < 1e-30) continue;
      double f = aug[row][col] / aug[col][col];
      for (int k = col; k <= 5; k++) aug[row][k] -= f * aug[col][k];
    }
  double x[5];
  for (int r = 4; r >= 0; r--) {
    x[r] = aug[r][5];
    for (int c = r + 1; c < 5; c++) x[r] -= aug[r][c] * x[c];
    x[r] = (fabs(aug[r][r]) > 1e-30) ? x[r] / aug[r][r] : 0.0;
  }

  // C_mat[5][8]: combine piece coefs
  double C_mat[5][8];
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 8; j++) C_mat[i][j] = 0.0;
  for (int i = 0; i < 32; i++) {
    int CRow = i % 4 + (i / 4) % 2;
    int CCol = i / 4;
    int coefRow = i % 8;
    int coefCol = i / 8;
    C_mat[CRow][CCol] = coefsOut[coefRow][coefCol];
  }

  // coefsSpline[2][4]: final piecewise polynomial coefficients
  double coefsSpline[2][4];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) coefsSpline[i][j] = 0.0;
  for (int j = 0; j < 8; j++) {
    int coefCol = j / 2;
    int coefRow = j % 2;
    for (int i = 0; i < 5; i++)
      coefsSpline[coefRow][coefCol] += C_mat[i][j] * x[i];
  }

  // Evaluate piecewise polynomial using Horner's method
  for (int i = 0; i < n; i++) {
    int sh = (i < br[1]) ? 0 : 1;
    yOut[i] = coefsSpline[sh][0];
  }
  for (int k = 1; k < 4; k++) {
    for (int j = 0; j < n; j++) {
      int sh = (j < br[1]) ? 0 : 1;
      yOut[j] = yOut[j] * (double)(j - br[1] * sh) + coefsSpline[sh][k];
    }
  }
}

// ============================================================================
// ── Feature 10: PD_PeriodicityWang_th0_01 ───────────────────────────────────
// Exact port: spline detrend (gpu_splinefit) + Pearson autocov
// (gpu_autocov_cov_mean).
// ============================================================================
__device__ __noinline__ static int gpu_PD_PeriodicityWang(const double* z,
                                                          int W, double* tmp) {
  // tmp (size W, from outer function's tmp_sort) used as spline scratch
  gpu_splinefit(z, W, tmp);

  // detrend: ySub = z - spline
  double ySub[C22_GPU_MAX_W];
  for (int i = 0; i < W; i++) ySub[i] = z[i] - tmp[i];

  // autocov of detrended signal (Pearson cov, matching CPU's autocov_lag)
  int acmax = (int)ceil((double)W / 3.0);
  double acf[C22_GPU_MAX_W];
  for (int lag = 1; lag <= acmax; lag++)
    acf[lag - 1] = gpu_autocov_cov_mean(ySub, W, lag);

  // find troughs and peaks in ACF
  double troughs[C22_GPU_MAX_W], peaks[C22_GPU_MAX_W];
  int nTroughs = 0, nPeaks = 0;
  for (int i = 1; i < acmax - 1; i++) {
    double sIn = acf[i] - acf[i - 1];
    double sOut = acf[i + 1] - acf[i];
    if (sIn < 0 && sOut > 0)
      troughs[nTroughs++] = i;
    else if (sIn > 0 && sOut < 0)
      peaks[nPeaks++] = i;
  }

  // first peak that satisfies: trough before it, peak-trough >= 0.01, peak > 0
  for (int i = 0; i < nPeaks; i++) {
    int iPeak = (int)peaks[i];
    double thePeak = acf[iPeak];
    int j = -1;
    while (j + 1 < nTroughs && (int)troughs[j + 1] < iPeak) j++;
    if (j < 0) continue;
    int iTrough = (int)troughs[j];
    double theTrough = acf[iTrough];
    if (thePeak - theTrough < 0.01) continue;
    if (thePeak < 0) continue;
    return iPeak;
  }
  return 0;
}

// ============================================================================
// ── Feature 11: CO_Embed2_Dist_tau_d_expfit_meandiff ─────────────────────────
// ============================================================================
__device__ static double gpu_CO_Embed2_Dist_expfit(const double* z, int W,
                                                   int tau) {
  if (tau <= 0) tau = 1;
  if (tau > W / 10) tau = W / 10;
  if (tau <= 0) tau = 1;

  int sz = W - tau - 1;
  if (sz <= 0) return 0.0;

  // distances in 2D embedding
  double d[C22_GPU_MAX_W];
  for (int i = 0; i < sz; i++) {
    double dx = z[i + 1] - z[i], dy = z[i + tau] - z[i + tau + 1];
    d[i] = sqrt(dx * dx + dy * dy);
  }
  double l = gpu_mean(d, sz);
  if (l < 1e-30) return 0.0;

  // histogram (auto bins)
  double mn = gpu_min(d, sz), mx = gpu_max(d, sz);
  double sd = gpu_stddev(d, sz);
  if (sd < 0.001) return 0.0;
  int nBins = (int)ceil((mx - mn) / (3.5 * sd / pow((double)sz, 1.0 / 3.0)));
  if (nBins <= 0) return 0.0;
  if (nBins > 100) nBins = 100;
  double binStep = (mx - mn) / nBins;

  double binCounts[100] = {};
  double binEdges[101];
  for (int b = 0; b <= nBins; b++) binEdges[b] = mn + b * binStep;
  for (int i = 0; i < sz; i++) {
    int b = (int)((d[i] - mn) / binStep);
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    binCounts[b]++;
  }
  for (int b = 0; b < nBins; b++) binCounts[b] /= sz;

  double acc = 0.0;
  for (int b = 0; b < nBins; b++) {
    double mid = (binEdges[b] + binEdges[b + 1]) * 0.5;
    double expf = exp(-mid / l) / l;
    if (expf < 0) expf = 0;
    acc += fabs(binCounts[b] - expf);
  }
  return acc / nBins;
}

// ============================================================================
// ── Feature 12: IN_AutoMutualInfoStats_40_gaussian_fmmi ──────────────────────
// ============================================================================
__device__ static double gpu_IN_AutoMutualInfoStats(const double* z, int W) {
  int tau = 40;
  if (tau > (int)ceil((double)W / 2)) tau = (int)ceil((double)W / 2);
  double ami[40];
  for (int i = 0; i < tau; i++) {
    // pycatch22 uses autocorr_lag() = Pearson corr of two sub-slices,
    // NOT the global sum/sum² form used by co_autocorrs().
    double ac = gpu_autocorr_pearson_lag(z, W, i + 1);
    double ac2 = ac * ac;
    ami[i] = (ac2 >= 1.0) ? 0.0 : -0.5 * log(1.0 - ac2);
  }
  for (int i = 1; i < tau - 1; i++)
    if (ami[i] < ami[i - 1] && ami[i] < ami[i + 1]) return (double)i;
  return (double)tau;
}

// ============================================================================
// ── Feature 13: FC_LocalSimple_mean1_tauresrat ───────────────────────────────
// ============================================================================
__device__ static double gpu_FC_LocalSimple_mean_tauresrat(const double* z,
                                                           int W, int train,
                                                           const double* ac) {
  // residuals of mean-1 forecast
  double res[C22_GPU_MAX_W];
  for (int i = 0; i < W - train; i++) {
    double yest = 0;
    for (int j = 0; j < train; j++) yest += z[i + j];
    yest /= train;
    res[i] = z[i + train] - yest;
  }
  int resSize = W - train;

  // first zero crossings of res and y
  double acRes[C22_GPU_MAX_W];
  gpu_co_autocorrs(res, resSize, acRes, resSize);
  int resAC1Z = gpu_co_firstzero(res, resSize, acRes, resSize);
  int yAC1Z = gpu_co_firstzero(z, W, ac, W);
  if (yAC1Z == 0) return 0.0;
  return (double)resAC1Z / (double)yAC1Z;
}

// ============================================================================
// ── Feature 14/15: DN_OutlierInclude ─────────────────────────────────────────
// ============================================================================
__device__ __noinline__ static double gpu_DN_OutlierInclude(
    const double* z, int W, double sign,
    double* r,          // W doubles
    double* msDti1,     // nThresh
    double* msDti3,     // nThresh
    double* msDti4,     // nThresh
    double* med_tmp) {  // W doubles
  double inc = 0.01;
  // check constant
  int constant = 1;
  for (int i = 0; i < W; i++)
    if (z[i] != z[0]) {
      constant = 0;
      break;
    }
  if (constant) return 0.0;

  // apply sign
  double yW[C22_GPU_MAX_W];
  int tot = 0;
  for (int i = 0; i < W; i++) {
    yW[i] = sign * z[i];
    if (yW[i] >= 0) tot++;
  }
  double maxVal = gpu_max(yW, W);
  if (maxVal < inc) return 0.0;

  int nThresh = (int)(maxVal / inc) + 1;
  if (nThresh > C22_GPU_MAX_NTHRESH) nThresh = C22_GPU_MAX_NTHRESH;

  double Dt_exc[C22_GPU_MAX_W];

  for (int j = 0; j < nThresh; j++) {
    int highSize = 0;
    for (int i = 0; i < W; i++)
      if (yW[i] >= j * inc) r[highSize++] = (double)(i + 1);
    if (highSize <= 1) {
      msDti1[j] = 0;
      msDti3[j] = 0;
      msDti4[j] = 0;
      continue;
    }
    for (int i = 0; i < highSize - 1; i++) Dt_exc[i] = r[i + 1] - r[i];
    msDti1[j] = gpu_mean(Dt_exc, highSize - 1);
    msDti3[j] = (double)(highSize - 1) * 100.0 / tot;
    msDti4[j] = gpu_median(r, highSize, med_tmp) / ((double)W / 2.0) - 1.0;
  }

  int trimthr = 2, mj = 0, fbi = nThresh - 1;
  for (int i = 0; i < nThresh; i++) {
    if (msDti3[i] > trimthr) mj = i;
    if (isnan(msDti1[nThresh - 1 - i])) fbi = nThresh - 1 - i;
  }
  int trimLimit = mj < fbi ? mj : fbi;
  return gpu_median(msDti4, trimLimit + 1, med_tmp);
}

// ============================================================================
// ── Features 16+21: SP_Summaries_welch_rect ──────────────────────────────────
// Returns area_5_1 in [0] and centroid in [1]
// ============================================================================
__device__ __noinline__ static void gpu_SP_Summaries_welch_rect(
    const double* z, int W, double* area_out, double* centroid_out,
    double* Fre,    // SP_NFFT
    double* Fim) {  // SP_NFFT
  const double PI = 3.14159265358979323846;
  int NFFT = gpu_nextpow2(W);
  if (NFFT > C22_GPU_SP_NFFT) NFFT = C22_GPU_SP_NFFT;

  double m = gpu_mean(z, W);
  // KMU = k * ||w||²  (Welch normalisation factor)
  // k=1 window (floor(W/(W/2))-1=1), rectangular window ||w||² = W.
  // Bug fix: was NFFT*NFFT which is 64²=4096 vs correct W=50 → 82× error.
  double KMU = (double)W;

  // Single Welch window (k=1 for W≈NFFT)
  double P[C22_GPU_SP_NFFT] = {};
  for (int i = 0; i < NFFT; i++) {
    Fre[i] = (i < W) ? z[i] - m : 0.0;
    Fim[i] = 0.0;
  }
  gpu_fft(Fre, Fim, NFFT);
  for (int i = 0; i < NFFT; i++) P[i] = Fre[i] * Fre[i] + Fim[i] * Fim[i];

  // one-sided spectrum with dt=1, df=1/NFFT
  double dt = 1.0, df = 1.0 / NFFT;
  int Nout = NFFT / 2 + 1;
  double Pxx[C22_GPU_SP_NFFT / 2 + 1], f_arr[C22_GPU_SP_NFFT / 2 + 1];
  for (int i = 0; i < Nout; i++) {
    Pxx[i] = P[i] / KMU * dt;
    if (i > 0 && i < Nout - 1) Pxx[i] *= 2.0;
    f_arr[i] = (double)i * df;
  }

  // angular freq and spectrum
  double w_arr[C22_GPU_SP_NFFT / 2 + 1], Sw[C22_GPU_SP_NFFT / 2 + 1];
  for (int i = 0; i < Nout; i++) {
    w_arr[i] = 2.0 * PI * f_arr[i];
    Sw[i] = Pxx[i] / (2.0 * PI);
    if (isinf(Sw[i])) {
      *area_out = 0;
      *centroid_out = 0;
      return;
    }
  }
  double dw = (Nout > 1) ? w_arr[1] - w_arr[0] : 1.0;

  // cumsum of Sw
  double csS[C22_GPU_SP_NFFT / 2 + 1];
  csS[0] = Sw[0];
  for (int i = 1; i < Nout; i++) csS[i] = csS[i - 1] + Sw[i];

  // area_5_1: integral over first fifth of spectrum
  double area = 0;
  for (int i = 0; i < Nout / 5; i++) area += Sw[i];
  *area_out = area * dw;

  // centroid: freq where cumsum exceeds 50%
  double threshold = csS[Nout - 1] * 0.5;
  *centroid_out = 0;
  for (int i = 0; i < Nout; i++)
    if (csS[i] > threshold) {
      *centroid_out = w_arr[i];
      break;
    }
}

// ============================================================================
// ── Feature 17: SB_BinaryStats_diff_longstretch0 ─────────────────────────────
// ============================================================================
__device__ static double gpu_SB_BinaryStats_diff_longstretch0(const double* z,
                                                              int W) {
  int maxStr = 0, last1 = 0;
  for (int i = 0; i < W - 1; i++) {
    int bin = (z[i + 1] - z[i] < 0) ? 0 : 1;
    if (bin == 1 || i == W - 2) {
      int stretch = i - last1;
      if (stretch > maxStr) maxStr = stretch;
      last1 = i;
    }
  }
  return (double)maxStr;
}

// ============================================================================
// ── Feature 18: SB_MotifThree_quantile_hh ────────────────────────────────────
// ============================================================================
__device__ static double gpu_SB_MotifThree_quantile_hh(const double* z, int W,
                                                       double* tmp_sort) {
  const int G = 3;
  int yt[C22_GPU_MAX_W];
  gpu_coarsegrain_quantile(z, W, G, yt, tmp_sort);

  // word frequencies (length-2 transitions)
  double out2[3][3] = {{}};
  for (int j = 0; j < W - 1; j++) {
    int a = yt[j] - 1, b = yt[j + 1] - 1;
    if (a >= 0 && a < G && b >= 0 && b < G) out2[a][b] += 1.0;
  }
  for (int a = 0; a < G; a++)
    for (int b = 0; b < G; b++) out2[a][b] /= (W - 1);

  // entropy of each row
  double hh = 0.0;
  for (int a = 0; a < G; a++) {
    for (int b = 0; b < G; b++) {
      if (out2[a][b] > 0) hh += out2[a][b] * log(out2[a][b]);
    }
  }
  return -hh;
}

// ============================================================================
// ── Internal: breakpoint regression for SC_FluctAnal ─────────────────────────
// Finds the optimal two-segment linear fit breakpoint on log-log data.
// Returns (firstMinInd + 1) / ntt  — matching pycatch22 indexing exactly.
// sserr[k] = norm(residuals_seg1) + norm(residuals_seg2)   (sum of L2 norms)
// ============================================================================
__device__ static double gpu_sc_fluct_breakpoint(const double* logtt,
                                                  const double* logFF, int ntt,
                                                  double* sserr) {
  int minPoints = 6;
  int nsserr = ntt - 2 * minPoints + 1;
  if (nsserr <= 0) return 0.0;

  double bestSS = DBL_MAX;
  double firstMinInd = 0.0;

  for (int i = minPoints; i < ntt - minPoints + 1; i++) {
    double m1, b1, m2, b2;
    gpu_linreg(logtt, logFF, i, &m1, &b1);
    gpu_linreg(logtt + (i - 1), logFF + (i - 1), ntt - i + 1, &m2, &b2);

    // Compute L2 norm of residuals for first segment
    double ss1 = 0.0;
    for (int j = 0; j < i; j++) {
      double e = logtt[j] * m1 + b1 - logFF[j];
      ss1 += e * e;
    }

    // Compute L2 norm of residuals for second segment
    double ss2 = 0.0;
    for (int j = 0; j < ntt - i + 1; j++) {
      double e = logtt[j + i - 1] * m2 + b2 - logFF[j + i - 1];
      ss2 += e * e;
    }

    // pycatch22 uses norm_(buffer, n) = sqrt(sum(e²)), then adds the two norms
    double sval = sqrt(ss1) + sqrt(ss2);
    sserr[i - minPoints] = sval;

    if (sval < bestSS) {
      bestSS = sval;
      // pycatch22: firstMinInd = (sserr_idx) + minPoints - 1
      //          = (i - minPoints) + minPoints - 1  =  i - 1
      firstMinInd = (double)(i - 1);
    }
  }
  return (firstMinInd + 1.0) / (double)ntt;
}

// ============================================================================
// ── Features 19+20: SC_FluctAnal ─────────────────────────────────────────────
// rsrangefit uses lag=1;  dfa uses lag=2  (must match pycatch22 calling
// convention in SC_FluctAnal.c).
// ============================================================================
__device__ __noinline__ static void gpu_SC_FluctAnal(
    const double* z, int W, double* rsrange_out, double* dfa_out,
    double* yCS,    // W doubles
    double* xReg,   // W/2 doubles
    double* logtt,  // 50 doubles
    double* logFF,  // 50 doubles
    double* sserr,  // 50 doubles
    double* buf) {  // W/2 doubles
  // ---- Shared: log-spaced tau vector (independent of lag) ----
  double linLow = log(5.0);
  // pycatch22 uses integer division: linHigh = log(size/2)
  double linHigh = log((double)(W / 2));
  int nTauSteps = 50;
  double tauStep = (linHigh - linLow) / (nTauSteps - 1);
  int tau[50];
  for (int i = 0; i < nTauSteps; i++)
    tau[i] = (int)round(exp(linLow + i * tauStep));

  // deduplicate
  int nTau = nTauSteps;
  for (int i = 0; i < nTau - 1;) {
    if (tau[i] == tau[i + 1]) {
      for (int j = i + 1; j < nTau - 1; j++) tau[j] = tau[j + 1];
      nTau--;
    } else
      i++;
  }
  if (nTau < 12) {
    *rsrange_out = 0;
    *dfa_out = 0;
    return;
  }

  int maxTau = tau[nTau - 1];
  for (int i = 0; i < maxTau; i++) xReg[i] = (double)(i + 1);

  // precompute log(tau)
  for (int i = 0; i < nTau; i++) logtt[i] = log((double)tau[i]);

  // ---- rsrangefit (lag = 1) ----
  {
    int lag = 1;
    int sizeCS = W / lag;
    yCS[0] = z[0];
    for (int i = 0; i < sizeCS - 1; i++)
      yCS[i + 1] = yCS[i] + z[(i + 1) * lag];

    double FArr[50] = {};
    for (int ti = 0; ti < nTau; ti++) {
      int t = tau[ti];
      int nBuf = sizeCS / t;
      if (nBuf == 0) continue;
      FArr[ti] = 0;
      for (int j = 0; j < nBuf; j++) {
        double m_lr, b_lr;
        gpu_linreg(xReg, yCS + j * t, t, &m_lr, &b_lr);
        double rmin = DBL_MAX, rmax = -DBL_MAX;
        for (int k = 0; k < t; k++) {
          buf[k] = yCS[j * t + k] - (m_lr * (k + 1) + b_lr);
          if (buf[k] < rmin) rmin = buf[k];
          if (buf[k] > rmax) rmax = buf[k];
        }
        FArr[ti] += (rmax - rmin) * (rmax - rmin);
      }
      FArr[ti] = sqrt(FArr[ti] / nBuf);
    }

    for (int i = 0; i < nTau; i++)
      logFF[i] = (FArr[i] > 0) ? log(FArr[i]) : -99.0;

    *rsrange_out = gpu_sc_fluct_breakpoint(logtt, logFF, nTau, sserr);
  }

  // ---- dfa (lag = 2) ----
  {
    int lag = 2;
    int sizeCS = W / lag;
    yCS[0] = z[0];
    for (int i = 0; i < sizeCS - 1; i++)
      yCS[i + 1] = yCS[i] + z[(i + 1) * lag];

    double FArr[50] = {};
    for (int ti = 0; ti < nTau; ti++) {
      int t = tau[ti];
      int nBuf = sizeCS / t;
      if (nBuf == 0) continue;
      FArr[ti] = 0;
      for (int j = 0; j < nBuf; j++) {
        double m_lr, b_lr;
        gpu_linreg(xReg, yCS + j * t, t, &m_lr, &b_lr);
        double sse = 0;
        for (int k = 0; k < t; k++) {
          double r = yCS[j * t + k] - (m_lr * (k + 1) + b_lr);
          sse += r * r;
        }
        FArr[ti] += sse;
      }
      FArr[ti] = sqrt(FArr[ti] / (nBuf * t));
    }

    for (int i = 0; i < nTau; i++)
      logFF[i] = (FArr[i] > 0) ? log(FArr[i]) : -99.0;

    *dfa_out = gpu_sc_fluct_breakpoint(logtt, logFF, nTau, sserr);
  }
}

// ============================================================================
// ── Feature 22: FC_LocalSimple_mean3_stderr ──────────────────────────────────
// ============================================================================
__device__ static double gpu_FC_LocalSimple_mean3_stderr(const double* z,
                                                         int W) {
  const int train = 3;
  double res[C22_GPU_MAX_W];
  for (int i = 0; i < W - train; i++) {
    double yest = (z[i] + z[i + 1] + z[i + 2]) / 3.0;
    res[i] = z[i + train] - yest;
  }
  return gpu_stddev(res, W - train);
}

// ============================================================================
// ── Master device function: compute all 22 features for one subsequence ──────
// ============================================================================
__device__ static void c22_compute_one_subsequence(const double* sub, int W,
                                                   double* feats) {
  // ── Workspace allocation (all local = L1-cached) ──────────────────────────
  double z[C22_GPU_MAX_W];       // z-scored subsequence
  double ac[C22_GPU_MAX_W + 1];  // autocorrelations, lags 0..W

  // For SC_FluctAnal
  double sc_yCS[C22_GPU_MAX_W];
  double sc_xReg[C22_GPU_MAX_W / 2];
  double sc_logtt[50], sc_logFF[50], sc_sserr[50];
  double sc_buf[C22_GPU_MAX_W / 2];

  // For SP_Summaries FFT
  double sp_Fre[C22_GPU_SP_NFFT], sp_Fim[C22_GPU_SP_NFFT];

  // For DN_OutlierInclude (largest workspace)
  double oi_r[C22_GPU_MAX_W];
  double oi_msDti1[C22_GPU_MAX_NTHRESH];
  double oi_msDti3[C22_GPU_MAX_NTHRESH];
  double oi_msDti4[C22_GPU_MAX_NTHRESH];
  double oi_med[C22_GPU_MAX_W];

  // General sort scratch
  double tmp_sort[C22_GPU_MAX_W];

  // ── Z-score normalise ─────────────────────────────────────────────────────
  gpu_zscore(sub, W, z);

  // ── Autocorrelations (computed once, reused by multiple features) ─────────
  gpu_co_autocorrs(z, W, ac, W);
  int firstZero = gpu_co_firstzero(z, W, ac, W);

  // ── Feature 1: DN_HistogramMode_5 ────────────────────────────────────────
  feats[0] = gpu_DN_HistogramMode(z, W, 5);
  // ── Feature 2: DN_HistogramMode_10 ───────────────────────────────────────
  feats[1] = gpu_DN_HistogramMode(z, W, 10);
  // ── Feature 3: CO_f1ecac ─────────────────────────────────────────────────
  feats[2] = gpu_CO_f1ecac(z, W, ac);
  // ── Feature 4: CO_FirstMin_ac ────────────────────────────────────────────
  feats[3] = (double)gpu_CO_FirstMin_ac(ac, W);
  // ── Feature 5: CO_HistogramAMI_even_2_5 ──────────────────────────────────
  feats[4] = gpu_CO_HistogramAMI_even_2_5(z, W);
  // ── Feature 6: CO_trev_1_num ─────────────────────────────────────────────
  feats[5] = gpu_CO_trev_1_num(z, W);
  // ── Feature 7: MD_hrv_classic_pnn40 ──────────────────────────────────────
  feats[6] = gpu_MD_hrv_pnn40(z, W);
  // ── Feature 8: SB_BinaryStats_mean_longstretch1 ───────────────────────────
  feats[7] = gpu_SB_BinaryStats_mean_longstretch1(z, W);
  // ── Feature 9: SB_TransitionMatrix_3ac_sumdiagcov ─────────────────────────
  feats[8] = gpu_SB_TransitionMatrix_3ac_sumdiagcov(z, W, ac, tmp_sort);
  // ── Feature 10: PD_PeriodicityWang_th0_01 ────────────────────────────────
  feats[9] = (double)gpu_PD_PeriodicityWang(z, W, tmp_sort);
  // ── Feature 11: CO_Embed2_Dist_tau_d_expfit_meandiff ─────────────────────
  feats[10] = gpu_CO_Embed2_Dist_expfit(z, W, firstZero);
  // ── Feature 12: IN_AutoMutualInfoStats_40_gaussian_fmmi ───────────────────
  feats[11] = gpu_IN_AutoMutualInfoStats(z, W);
  // ── Feature 13: FC_LocalSimple_mean1_tauresrat ────────────────────────────
  feats[12] = gpu_FC_LocalSimple_mean_tauresrat(z, W, 1, ac);
  // ── Feature 14: DN_OutlierInclude_p_001_mdrmd ─────────────────────────────
  feats[13] = gpu_DN_OutlierInclude(z, W, 1.0, oi_r, oi_msDti1, oi_msDti3,
                                    oi_msDti4, oi_med);
  // ── Feature 15: DN_OutlierInclude_n_001_mdrmd ─────────────────────────────
  feats[14] = gpu_DN_OutlierInclude(z, W, -1.0, oi_r, oi_msDti1, oi_msDti3,
                                    oi_msDti4, oi_med);
  // ── Features 16+21: SP_Summaries (computed together via shared FFT) ───────
  {
    double area, centroid;
    gpu_SP_Summaries_welch_rect(z, W, &area, &centroid, sp_Fre, sp_Fim);
    feats[15] = area;      // SP_Summaries_welch_rect_area_5_1
    feats[20] = centroid;  // SP_Summaries_welch_rect_centroid
  }
  // ── Feature 17: SB_BinaryStats_diff_longstretch0 ─────────────────────────
  feats[16] = gpu_SB_BinaryStats_diff_longstretch0(z, W);
  // ── Feature 18: SB_MotifThree_quantile_hh ────────────────────────────────
  feats[17] = gpu_SB_MotifThree_quantile_hh(z, W, tmp_sort);
  // ── Features 19+20: SC_FluctAnal ─────────────────────────────────────────
  {
    double rsrange, dfa;
    gpu_SC_FluctAnal(z, W, &rsrange, &dfa, sc_yCS, sc_xReg,
                     sc_logtt, sc_logFF, sc_sserr, sc_buf);
    feats[18] = rsrange;  // SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
    feats[19] = dfa;      // SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
  }
  // ── Feature 22: FC_LocalSimple_mean3_stderr ───────────────────────────────
  feats[21] = gpu_FC_LocalSimple_mean3_stderr(z, W);

  // Replace any NaN/Inf with 0 (matches CPU behaviour in c22_features.cpp)
  for (int f = 0; f < C22_GPU_NUM_FEAT; f++)
    if (!isfinite(feats[f])) feats[f] = 0.0;
}

// ============================================================================
// ── Global kernel: one thread per subsequence ────────────────────────────────
// Grid: ceil(N / C22_GPU_BLOCK_SIZE) × 1
// Block: C22_GPU_BLOCK_SIZE × 1
// ============================================================================
__global__ void c22_compute_features_kernel(
    const double* __restrict__ timeseries,  // raw time series (ts_length)
    int ts_length, int window,
    int N,                                // = ts_length - window + 1
    double* __restrict__ features_out) {  // N × 22 (row-major)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  if (window > C22_GPU_MAX_W) return;  // skip (handled by CPU fallback)

  const double* sub = timeseries + idx;  // subsequence starts here
  double* out = features_out + (size_t)idx * C22_GPU_NUM_FEAT;
  c22_compute_one_subsequence(sub, window, out);
}

// ============================================================================
// ── Host launch wrapper
// ───────────────────────────────────────────────────────
// ============================================================================
extern "C" {

// Compute all N×22 features on GPU for the given timeseries.
// Returns true on success; false if window > C22_GPU_MAX_W (caller uses CPU).
bool c22_compute_features_gpu_launch(const double* h_ts, int ts_length,
                                     int window, int N, double* h_features_out,
                                     int gpu_id) {
  if (window > C22_GPU_MAX_W) return false;

  cudaSetDevice(gpu_id);

  // Increase stack size to accommodate large per-thread local arrays
  cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024);  // 128 KB per thread

  size_t ts_bytes = (size_t)ts_length * sizeof(double);
  size_t feat_bytes = (size_t)N * C22_GPU_NUM_FEAT * sizeof(double);

  double* d_ts = nullptr;
  double* d_features = nullptr;

  if (cudaMalloc(&d_ts, ts_bytes) != cudaSuccess) return false;
  if (cudaMalloc(&d_features, feat_bytes) != cudaSuccess) {
    cudaFree(d_ts);
    return false;
  }

  cudaMemcpy(d_ts, h_ts, ts_bytes, cudaMemcpyHostToDevice);

  int blocks = (N + C22_GPU_BLOCK_SIZE - 1) / C22_GPU_BLOCK_SIZE;
  c22_compute_features_kernel<<<blocks, C22_GPU_BLOCK_SIZE>>>(
      d_ts, ts_length, window, N, d_features);

  cudaError_t err = cudaDeviceSynchronize();
  bool ok = (err == cudaSuccess);
  if (!ok)
    fprintf(stderr, "c22_compute_features_kernel error: %s\n",
            cudaGetErrorString(err));

  if (ok)
    cudaMemcpy(h_features_out, d_features, feat_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_ts);
  cudaFree(d_features);
  return ok;
}

}  // extern "C"

#endif  // _HAS_CUDA_
