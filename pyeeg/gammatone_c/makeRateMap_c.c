/*
 *=============================================================
%MAKERATEMAP_C: A C implementation of MPC's makeRateMap matlab code
%--------------------------------------------------
%  ratemap = makeRateMap_c(x,fs,lowcf,highcf,numchans,frameshift,ti,compression)
% 
%  x           input signal
%  fs          sampling frequency in Hz (8000)
%  lowcf       centre frequency of lowest filter in Hz (50)
%  highcf      centre frequency of highest filter in Hz (3500)
%  numchans    number of channels in filterbank (32)
%  frameshift  interval between successive frames in ms (10)
%  ti          temporal integration in ms (8)
%  compression type of compression ['cuberoot','log','none'] ('cuberoot')
%
%  e.g. ratemap = makeRateMap_c(x,8000,50,3850,32,10);
%
%
%  For more detail on this implementation, see
%  https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/
%
%  Ning Ma, University of Sheffield
%  n.ma@dcs.shef.ac.uk, 08 Dec 2005
%=============================================================
*/
// filepath: c:\Users\hugwei\Documents\gammatone\makeRateMap_c.c
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

// Define constants
#define BW_CORRECTION   1.019
#define VERY_SMALL_NUMBER  1e-200
#define LOG10_VERY_SMALL_NUMBER  -200

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif

// Utility functions
#define getMax(x,y)     ((x)>(y)?(x):(y))
#define getRound(x)     ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#define erb(x)          (24.7*(4.37e-3*(x)+1.0))
#define HzToErbRate(x)  (21.4*log10(4.37e-3*(x)+1.0))
#define ErbRateToHz(x)  ((pow(10.0,((x)/21.4))-1.0)/4.37e-3)

// Function to create a double matrix
double* createDoubleMatrix(int rows, int cols) {
    return (double*)calloc(rows * cols, sizeof(double));
}

// Function to get the pointer to the data of a double matrix
double* getPr(double* matrix) {
    return matrix;
}

// Function to free allocated memory
void freeMatrix(double* matrix) {
    free(matrix);
}

// Main function
void makeRateMap(double* x, int nsamples, int fs, double lowcf, double highcf, int numchans, double frameshift, double ti, const char* compression, double* ratemap) {
    double *senv;
    int i, j, chan, frameshift_samples, numframes, nsamples_padded;
    double lowErb, highErb, spaceErb, cf;
    double a, tpt, tptbw, gain, intdecay, intgain, sumEnv;
    double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i;
    double a1, a2, a3, a4, a5, cs, sn, u0r, u0i;
    double senv1, oldcs, oldsn, coscf, sincf;

    frameshift_samples = getRound(frameshift * fs / 1000);
    numframes = (int)ceil((double)nsamples / (double)frameshift_samples);
    nsamples_padded = numframes * frameshift_samples;

    lowErb = HzToErbRate(lowcf);
    highErb = HzToErbRate(highcf);
    spaceErb = (numchans > 1) ? (highErb - lowErb) / (numchans - 1) : 0.0;

    senv = (double*)calloc(nsamples_padded, sizeof(double));

    tpt = 2 * M_PI / fs;
    intdecay = exp(-(1000.0 / (fs * ti)));
    intgain = 1 - intdecay;

    for (chan = 0; chan < numchans; chan++) {
        cf = ErbRateToHz(lowErb + chan * spaceErb);
        tptbw = tpt * erb(cf) * BW_CORRECTION;
        a = exp(-tptbw);
        gain = (tptbw * tptbw * tptbw * tptbw) / 3;

        a1 = 4.0 * a; a2 = -6.0 * a * a; a3 = 4.0 * a * a * a; a4 = -a * a * a * a; a5 = a * a;

        p0r = p1r = p2r = p3r = p4r = p0i = p1i = p2i = p3i = p4i = 0.0;
        senv1 = 0.0;

        coscf = cos(tpt * cf);
        sincf = sin(tpt * cf);
        cs = 1; sn = 0;

        for (i = 0; i < nsamples; i++) {
            p0r = cs * x[i] + a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r;
            p0i = sn * x[i] + a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i;

            if (fabs(p0r) < VERY_SMALL_NUMBER) p0r = 0.0F;
            if (fabs(p0i) < VERY_SMALL_NUMBER) p0i = 0.0F;

            u0r = p0r + a1 * p1r + a5 * p2r;
            u0i = p0i + a1 * p1i + a5 * p2i;

            p4r = p3r; p3r = p2r; p2r = p1r; p1r = p0r;
            p4i = p3i; p3i = p2i; p2i = p1i; p1i = p0i;

            senv1 = senv[i] = sqrt(u0r * u0r + u0i * u0i) * gain + intdecay * senv1;

            cs = (oldcs = cs) * coscf + (oldsn = sn) * sincf;
            sn = oldsn * coscf - oldcs * sincf;
        }

        for (i = nsamples; i < nsamples_padded; i++) {
            p0r = a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r;
            p0i = a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i;

            u0r = p0r + a1 * p1r + a5 * p2r;
            u0i = p0i + a1 * p1i + a5 * p2i;

            p4r = p3r; p3r = p2r; p2r = p1r; p1r = p0r;
            p4i = p3i; p3i = p2i; p2i = p1i; p1i = p0i;

            senv1 = senv[i] = sqrt(u0r * u0r + u0i * u0i) * gain + intdecay * senv1;
        }

        for (j = 0; j < numframes; j++) {
            sumEnv = 0.0;
            for (i = j * frameshift_samples; i < (j + 1) * frameshift_samples; i++) {
                sumEnv += senv[i];
            }
            ratemap[chan + numchans * j] = intgain * sumEnv / frameshift_samples;
        }
    }

    if (strcmp(compression, "cuberoot") == 0) {
        for (i = 0; i < numchans * numframes; i++) {
            ratemap[i] = pow(ratemap[i], 0.3);
        }
    } else if (strcmp(compression, "log") == 0) {
        for (i = 0; i < numchans * numframes; i++) {
            if (ratemap[i] > VERY_SMALL_NUMBER) {
                ratemap[i] = log10(ratemap[i]);
            } else {
                ratemap[i] = LOG10_VERY_SMALL_NUMBER;
            }
        }
    }

    free(senv);
}

// Python wrapper for makeRateMap
static PyObject* py_makeRateMap(PyObject* self, PyObject* args) {
    PyObject *x_obj, *ratemap_obj;
    int nsamples, fs, numchans;
    double lowcf, highcf, frameshift, ti;
    const char* compression;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "OiiddiddsO", &x_obj, &nsamples, &fs, &lowcf, &highcf, &numchans, &frameshift, &ti, &compression, &ratemap_obj)) {
        return NULL;
    }

    // Convert input signal to C array
    double* x = (double*)PyArray_DATA((PyArrayObject*)x_obj);

    // Allocate output array
    double* ratemap = (double*)PyArray_DATA((PyArrayObject*)ratemap_obj);

    // Call the C function
    makeRateMap(x, nsamples, fs, lowcf, highcf, numchans, frameshift, ti, compression, ratemap);

    Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef MakeRateMapMethods[] = {
    {"makeRateMap", py_makeRateMap, METH_VARARGS, "Compute the rate map using a gammatone filterbank."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef makeRateMap_module = {
    PyModuleDef_HEAD_INIT,
    "makeRateMap_c",
    "Rate map computation implemented in C.",
    -1,
    MakeRateMapMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_makeRateMap_c(void) {
    import_array(); // Required for NumPy
    return PyModule_Create(&makeRateMap_module);
}