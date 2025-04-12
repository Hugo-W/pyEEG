/*
 *=========================================================================
 * An efficient C implementation of the 4th order gammatone filter
 *-------------------------------------------------------------------------
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *-------------------------------------------------------------------------
%
%  [bm, env, instp, instf] = gammatone_c(x, fs, cf, hrect) 
%
%  x     - input signal
%  fs    - sampling frequency (Hz)
%  cf    - centre frequency of the filter (Hz)
%  hrect - half-wave rectifying if hrect = 1 (default 0)
%
%  bm    - basilar membrane displacement
%  env   - instantaneous envelope
%  instp - instantaneous phase (unwrapped radian)
%  instf - instantaneous frequency (Hz)
%
%
%  The gammatone filter is commonly used in models of the auditory system.
%  The algorithm is based on Martin Cooke's Ph.D work (Cooke, 1993) using 
%  the base-band impulse invariant transformation. This implementation is 
%  highly efficient in that a mathematical rearrangement is used to 
%  significantly reduce the cost of computing complex exponentials. For 
%  more detail on this implementation see
%  https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/
%
%  Ning Ma, University of Sheffield
%  n.ma@dcs.shef.ac.uk, 09 Mar 2006
% 
 * CHANGES:
 * 2012-05-30 Ning Ma <n.ma@dcs.shef.ac.uk>
 *   Fixed a typo in the implementation (a5). The typo does not make a lot
 *   of difference to the response. Thanks to Vijay Parsa for reporting
 *   the problem.
 *
 * 2010-02-01 Ning Ma <n.ma@dcs.shef.ac.uk>
 *   Clip very small filter coefficients to zero in order to prevent
 *   gradual underflow. Arithmetic operations may become very slow with
 *   subnormal numbers (those smaller than the minimum positive normal
 *   value, 2.225e-308 in double precision). This could happen if the 
 *   input signal cotains many zeros (e.g. impulse responses). Thanks to
 *   John Culling for reporting the problem.
 *=========================================================================
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Define constants
#define BW_CORRECTION      1.0190
#define VERY_SMALL_NUMBER  1e-200
#ifndef M_PI
#define M_PI               3.14159265358979323846
#endif

// Utility functions
#define myMax(x,y)     ( ( x ) > ( y ) ? ( x ) : ( y ) )
#define myMod(x,y)     ( ( x ) - ( y ) * floor ( ( x ) / ( y ) ) )
#define erb(x)         ( 24.7 * ( 4.37e-3 * ( x ) + 1.0 ) )

// Function to create a double matrix
double* createDoubleMatrix(int rows, int cols) {
   return (double*)calloc(rows * cols, sizeof(double));
}

// Function to free allocated memory
void freeMatrix(double* matrix) {
   free(matrix);
}

// Main gammatone function
void gammatone_c(double* x, int nsamples, int fs, double cf, int hrect, double* bm, double* env, double* instp, double* instf) {
   double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i;
   double a, tpt, tptbw, gain;
   double a1, a2, a3, a4, a5, u0r, u0i;
   double qcos, qsin, oldcs, coscf, sincf, oldphase, dp, dps;

   // Initialize variables
   oldphase = 0.0;
   tpt = (M_PI + M_PI) / fs;
   tptbw = tpt * erb(cf) * BW_CORRECTION;
   a = exp(-tptbw);
   gain = (tptbw * tptbw * tptbw * tptbw) / 3;
   a1 = 4.0 * a; a2 = -6.0 * a * a; a3 = 4.0 * a * a * a; a4 = -a * a * a * a; a5 = a * a;
   p0r = p1r = p2r = p3r = p4r = p0i = p1i = p2i = p3i = p4i = 0.0;
   coscf = cos(tpt * cf);
   sincf = sin(tpt * cf);
   qcos = 1; qsin = 0;

   for (int t = 0; t < nsamples; t++) {
       p0r = qcos * x[t] + a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r;
       p0i = qsin * x[t] + a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i;
       if (fabs(p0r) < VERY_SMALL_NUMBER) p0r = 0.0F;
       if (fabs(p0i) < VERY_SMALL_NUMBER) p0i = 0.0F;
       u0r = p0r + a1 * p1r + a5 * p2r;
       u0i = p0i + a1 * p1i + a5 * p2i;
       p4r = p3r; p3r = p2r; p2r = p1r; p1r = p0r;
       p4i = p3i; p3i = p2i; p2i = p1i; p1i = p0i;
       bm[t] = (u0r * qcos + u0i * qsin) * gain;
       if (hrect == 1 && bm[t] < 0) bm[t] = 0;
       if (env != NULL) env[t] = sqrt(u0r * u0r + u0i * u0i) * gain;
       if (instp != NULL) {
           instp[t] = atan2(u0i, u0r);
           dp = instp[t] - oldphase;
           if (fabs(dp) > M_PI) {
               dps = myMod(dp + M_PI, 2 * M_PI) - M_PI;
               if (dps == -M_PI && dp > 0) dps = M_PI;
               instp[t] = instp[t] + dps - dp;
           }
           oldphase = instp[t];
       }
       if (instf != NULL && t > 0) instf[t - 1] = cf + (instp[t] - instp[t - 1]) / tpt;
       qcos = coscf * (oldcs = qcos) + sincf * qsin;
       qsin = coscf * qsin - sincf * oldcs;
   }
   if (instf != NULL) instf[nsamples - 1] = cf;
}

// Wrapper function for Python
static PyObject* py_gammatone_c(PyObject* self, PyObject* args) {
    PyObject *x_obj, *bm_obj, *env_obj = NULL, *instp_obj = NULL, *instf_obj = NULL;
    int nsamples, fs, hrect;
    double cf;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "OiidO|OOO", &x_obj, &nsamples, &fs, &cf, &bm_obj, &env_obj, &instp_obj, &instf_obj)) {
        return NULL;
    }

    // Convert input signal to C array
    double* x = (double*)PyArray_DATA((PyArrayObject*)x_obj);

    // Allocate output arrays
    double* bm = (double*)PyArray_DATA((PyArrayObject*)bm_obj);
    double* env = env_obj ? (double*)PyArray_DATA((PyArrayObject*)env_obj) : NULL;
    double* instp = instp_obj ? (double*)PyArray_DATA((PyArrayObject*)instp_obj) : NULL;
    double* instf = instf_obj ? (double*)PyArray_DATA((PyArrayObject*)instf_obj) : NULL;

    // Call the C function
    gammatone_c(x, nsamples, fs, cf, hrect, bm, env, instp, instf);

    Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef GammatoneMethods[] = {
    {"gammatone_c", py_gammatone_c, METH_VARARGS, "Apply gammatone filter to a signal."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef gammatone_module = {
    PyModuleDef_HEAD_INIT,
    "gammatone_c",
    "Gammatone filter implemented in C.",
    -1,
    GammatoneMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_gammatone_c(void) {
    import_array(); // Required for NumPy
    return PyModule_Create(&gammatone_module);
}