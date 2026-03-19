#ifndef C22_AUTOCORR_OPT_H
#define C22_AUTOCORR_OPT_H

#ifdef __cplusplus
extern "C" {
#endif

double *c22_co_autocorrs(const double y[], int size);

int c22_co_firstzero_cached(const double *autocorrs, int size, int maxtau);
double c22_CO_f1ecac_cached(const double *autocorrs, int size);
int c22_CO_FirstMin_ac_cached(const double *autocorrs, int size);
double c22_CO_Embed2_Dist_tau_d_expfit_meandiff_cached(
    const double y[], int size, const double *autocorrs);
double c22_FC_LocalSimple_mean1_tauresrat_cached(
    const double y[], int size, const double *autocorrs);
double c22_SB_TransitionMatrix_3ac_sumdiagcov_cached(
    const double y[], int size, const double *autocorrs);

#ifdef __cplusplus
}
#endif

#endif
