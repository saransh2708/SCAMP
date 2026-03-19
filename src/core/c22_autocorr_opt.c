/*
 * c22_autocorr_opt.c — Cached-autocorrelation versions of catch22 features.
 *
 * Several catch22 features internally compute autocorrelation via
 * co_autocorrs() (FFT-based, O(W log W)). When computing all 22 features
 * for one subsequence, the original code calls co_autocorrs() 6 times
 * redundantly. This file provides _cached variants that accept a
 * pre-computed autocorrelation array, avoiding the redundant FFT work.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "CO_AutoCorr.h"
#include "SB_CoarseGrain.h"
#include "SB_TransitionMatrix.h"
#include "FC_LocalSimple.h"
#include "stats.h"
#include "histcounts.h"
#include "c22_autocorr_opt.h"

double *c22_co_autocorrs(const double y[], int size) {
    return co_autocorrs(y, size);
}

int c22_co_firstzero_cached(const double *autocorrs, int size, int maxtau) {
    int zerocrossind = 0;
    while (autocorrs[zerocrossind] > 0 && zerocrossind < maxtau) {
        zerocrossind += 1;
    }
    return zerocrossind;
}

double c22_CO_f1ecac_cached(const double *autocorrs, int size) {
    double thresh = 1.0 / exp(1);
    double out = (double)size;
    for (int i = 0; i < size - 2; i++) {
        if (autocorrs[i + 1] < thresh) {
            double m = autocorrs[i + 1] - autocorrs[i];
            double dy = thresh - autocorrs[i];
            double dx = dy / m;
            return ((double)i) + dx;
        }
    }
    return out;
}

int c22_CO_FirstMin_ac_cached(const double *autocorrs, int size) {
    int minInd = size;
    for (int i = 1; i < size - 1; i++) {
        if (autocorrs[i] < autocorrs[i - 1] && autocorrs[i] < autocorrs[i + 1]) {
            minInd = i;
            break;
        }
    }
    return minInd;
}

double c22_CO_Embed2_Dist_tau_d_expfit_meandiff_cached(
    const double y[], int size, const double *autocorrs) {
    for (int i = 0; i < size; i++) {
        if (isnan(y[i])) return NAN;
    }

    int tau = c22_co_firstzero_cached(autocorrs, size, size);
    if (tau > (double)size / 10) {
        tau = floor((double)size / 10);
    }

    double *d = malloc((size - tau) * sizeof(double));
    for (int i = 0; i < size - tau - 1; i++) {
        d[i] = sqrt((y[i + 1] - y[i]) * (y[i + 1] - y[i]) +
                     (y[i + tau] - y[i + tau + 1]) * (y[i + tau] - y[i + tau + 1]));
        if (isnan(d[i])) {
            free(d);
            return NAN;
        }
    }

    double l = mean(d, size - tau - 1);

    int nBins = num_bins_auto(d, size - tau - 1);
    if (nBins == 0) {
        free(d);
        return 0;
    }
    int *histCounts = malloc(nBins * sizeof(int));
    double *binEdges = malloc((nBins + 1) * sizeof(double));
    histcounts_preallocated(d, size - tau - 1, nBins, histCounts, binEdges);

    double *histCountsNorm = malloc(nBins * sizeof(double));
    for (int i = 0; i < nBins; i++) {
        histCountsNorm[i] = (double)histCounts[i] / (double)(size - tau - 1);
    }

    double *d_expfit_diff = malloc(nBins * sizeof(double));
    for (int i = 0; i < nBins; i++) {
        double expf = exp(-(binEdges[i] + binEdges[i + 1]) * 0.5 / l) / l;
        if (expf < 0) expf = 0;
        d_expfit_diff[i] = fabs(histCountsNorm[i] - expf);
    }

    double out = mean(d_expfit_diff, nBins);

    free(d);
    free(d_expfit_diff);
    free(binEdges);
    free(histCountsNorm);
    free(histCounts);
    return out;
}

double c22_FC_LocalSimple_mean1_tauresrat_cached(
    const double y[], int size, const double *autocorrs) {
    int train_length = 1;

    double *res = malloc((size - train_length) * sizeof(double));
    for (int i = 0; i < size - train_length; i++) {
        double yest = 0;
        for (int j = 0; j < train_length; j++) {
            yest += y[i + j];
        }
        yest /= train_length;
        res[i] = y[i + train_length] - yest;
    }

    /* resAC1stZ: residual is a new series, must compute its own autocorrelation */
    double resAC1stZ = co_firstzero(res, size - train_length, size - train_length);
    /* yAC1stZ: use pre-computed autocorrelation of the original series */
    double yAC1stZ = c22_co_firstzero_cached(autocorrs, size, size);
    double output = resAC1stZ / yAC1stZ;

    free(res);
    return output;
}

double c22_SB_TransitionMatrix_3ac_sumdiagcov_cached(
    const double y[], int size, const double *autocorrs) {
    int constant = 1;
    for (int i = 0; i < size; i++) {
        if (isnan(y[i])) return NAN;
        if (y[i] != y[0]) constant = 0;
    }
    if (constant) return NAN;

    const int numGroups = 3;
    int tau = c22_co_firstzero_cached(autocorrs, size, size);

    double *yFilt = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        yFilt[i] = y[i];
    }

    int nDown = (size - 1) / tau + 1;
    double *yDown = malloc(nDown * sizeof(double));
    for (int i = 0; i < nDown; i++) {
        yDown[i] = yFilt[i * tau];
    }

    int *yCG = malloc(nDown * sizeof(int));
    sb_coarsegrain(yDown, nDown, "quantile", numGroups, yCG);

    double T[3][3];
    for (int i = 0; i < numGroups; i++)
        for (int j = 0; j < numGroups; j++)
            T[i][j] = 0;

    for (int j = 0; j < nDown - 1; j++) {
        T[yCG[j] - 1][yCG[j + 1] - 1] += 1;
    }

    for (int i = 0; i < numGroups; i++)
        for (int j = 0; j < numGroups; j++)
            T[i][j] /= (nDown - 1);

    double column1[3] = {0}, column2[3] = {0}, column3[3] = {0};
    for (int i = 0; i < numGroups; i++) {
        column1[i] = T[i][0];
        column2[i] = T[i][1];
        column3[i] = T[i][2];
    }
    double *columns[3] = {column1, column2, column3};

    double sumdiagcov = 0;
    for (int i = 0; i < numGroups; i++) {
        sumdiagcov += cov(columns[i], columns[i], 3);
    }

    free(yFilt);
    free(yDown);
    free(yCG);
    return sumdiagcov;
}
