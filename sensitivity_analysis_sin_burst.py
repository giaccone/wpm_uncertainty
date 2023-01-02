# import common modules
import numpy as np
from progress.bar import Bar
# exposure modules
from exposure.icnirp import icnirp_filter
from exposure.modified.icnirp import icnirp_filter as icnirp_filter_proposed
from fourier.ffttools import fft_analysis
from exposure.pulsed import wpm_time, wpm_freq
# polynomial chaos expansion
import pce as pce
# waveform
from waveform import sin_burst

# Define filter
year = '2010'
receptor = 'occupational'
quantity = 'B'
Glim = 1000e-6


# function for the time domain analysis
def fun_time(x, year=year, receptor=receptor, quantity=quantity):

    def inner_fun(xx):
        # Define time domain waveform
        fr = xx[0]
        # Glim = xx[1]
        delay = xx[1]
        B, t = sin_burst(Glim, fr, delay=delay, n_period=11, shape=(0, 0, 80, 90, 100, 100, 100, 90, 80, 0, 0))

        # WP in time domain
        num, den = icnirp_filter(year, receptor, quantity, 'time')
        IW, _ = wpm_time(num, den, t, B, makeplot='n')

        return IW
    from joblib import Parallel, delayed
    y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))

    return np.array(y)


# function for the frequency domain analysis
def fun_freq(x, year=year, receptor=receptor, quantity=quantity):

    def inner_fun(xx):
        # Define time domain waveform
        fr = xx[0]
        # Glim = xx[1]
        delay = xx[1]
        B, t = sin_burst(Glim, fr, delay=delay, n_period=11, shape=(0, 0, 80, 90, 100, 100, 100, 90, 80, 0, 0))

        # Evaluate spectrum
        f, BFT = fft_analysis(t, B)

        # WP in frequency domain
        WF, phi = icnirp_filter(year, receptor, quantity, 'freq', f)
        IWf, _ = wpm_freq(WF, phi, f, BFT, makeplot='n')

        return IWf
    from joblib import Parallel, delayed
    y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))

    return np.array(y)


# function for the frequency domain analysis (proposed)
def fun_freq_new(x, year=year, receptor=receptor, quantity=quantity):

    def inner_fun(xx):
        # Define time domain waveform
        fr = xx[0]
        # Glim = xx[1]
        delay = xx[1]
        B, t = sin_burst(Glim, fr, delay=delay, n_period=11, shape=(0, 0, 80, 90, 100, 100, 100, 90, 80, 0, 0))

        # Evaluate spectrum
        f, BFT = fft_analysis(t, B)

        # WP in frequency domain (proposed)
        WFb, phib = icnirp_filter_proposed(year, receptor, quantity, 'freq', f)
        IWfb, _ = wpm_freq(WFb, phib, f, BFT, makeplot='n')

        return IWfb
    from joblib import Parallel, delayed
    y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))

    return np.array(y)


# PCE
orders = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
lev = 7
index = [[1], [2], [1, 2]]
distrib = ['n']*2
fr = 300
delay = 0.5/fr
param = [[fr, fr*1/100], [delay, delay*1/100]]


# variable used to summarize all results
summary = np.zeros((len(index) + 2, 3))

# Begin analysis ...
# time domain
S = np.zeros((len(index), len(orders)))
bar = Bar('Processing', max=len(orders))
for k, order in enumerate(orders):
    # generate PCE
    poly = pce.PolyChaos(order, distrib, param)

    # compute coefficients
    poly.spectral_projection(fun_time, lev, verbose='n')
    poly.norm_fit()
    sobol_index = poly.sobol(index)
    S[:, k] = np.array(sobol_index)
    bar.next()
bar.finish()

# print first line
first_line = "order & " + " & ".join([str(k) for k in orders]) + " \\\\"
print(first_line)
# print sobol index
for idx, sobol, in zip(index, S):
    format_str = "".join([str(ele) for ele in idx])
    format_str = "S" + format_str
    format_str = format_str  + " & {:.4f}"*len(sobol) + " \\\\"
    print(format_str.format(*sobol))

# scaling factor results to have EI=1 for time domain
scaling = 1 / poly.mu
# update summary
summary[:, 0] = np.concatenate((S[:,-1], [poly.mu * scaling, poly.sigma * scaling]))
# print mu and sigma
print("mu = {:.5f}   sigma = {:.5f}\n".format(poly.mu  * scaling, poly.sigma * scaling))
print("------------------------------------")
print("------------------------------------")

# frequency domain
S = np.zeros(( len(index), len(orders)))
bar = Bar('Processing', max=len(orders))
for k, order in enumerate(orders):
    # generate PCE
    poly = pce.PolyChaos(order, distrib, param)

    # compute coefficients
    poly.spectral_projection(fun_freq, lev, verbose='n')
    poly.norm_fit()
    sobol_index = poly.sobol(index)
    S[:, k] = np.array(sobol_index)
    bar.next()
bar.finish()

# print first line
first_line = "order & " + " & ".join([str(k) for k in orders]) + " \\\\"
print(first_line)
# print sobol index
for idx, sobol, in zip(index, S):
    format_str = "".join([str(ele) for ele in idx])
    format_str = "S" + format_str
    format_str = format_str  + " & {:.4f}"*len(sobol) + " \\\\"
    print(format_str.format(*sobol))

# update summary
summary[:, 1] = np.concatenate((S[:,-1], [poly.mu * scaling, poly.sigma * scaling]))
# print mu and sigma
print("mu = {:.5f}   sigma = {:.5f}\n".format(poly.mu  * scaling, poly.sigma * scaling))
print("------------------------------------")
print("------------------------------------")


# frequency domain
S = np.zeros(( len(index), len(orders)))
bar = Bar('Processing', max=len(orders))
for k, order in enumerate(orders):
    # generate PCE
    poly = pce.PolyChaos(order, distrib, param)

    # compute coefficients
    poly.spectral_projection(fun_freq_new, lev, verbose='n')
    poly.norm_fit()
    sobol_index = poly.sobol(index)
    S[:, k] = np.array(sobol_index)
    bar.next()
bar.finish()

# print first line
first_line = "order & " + " & ".join([str(k) for k in orders]) + " \\\\"
print(first_line)
# print sobol index
for idx, sobol, in zip(index, S):
    format_str = "".join([str(ele) for ele in idx])
    format_str = "S" + format_str
    format_str = format_str  + " & {:.4f}"*len(sobol) + " \\\\"
    print(format_str.format(*sobol))

# update summary
summary[:, 2] = np.concatenate((S[:,-1], [poly.mu * scaling, poly.sigma * scaling]))
# print mu and sigma
print("mu = {:.5f}   sigma = {:.5f}\n".format(poly.mu  * scaling, poly.sigma * scaling))
print("------------------------------------")
print("------------------------------------")

# print summary (in the form of LaTeX table)
for idx, row in zip(index, summary[:-2, :]):
    print("$S_{" + "".join([str(ele) for ele in idx]) + "}$ & ", end="")
    print("{:.3f} & {:.3f} & {:.3f} \\\\".format(*row*100))
for ele, row in zip(["\\mu", "\\sigma"], summary[-2:, :]):
    print("${}$ & {:.2f} & {:.2f} & {:.2f} \\\\".format(ele, *row))

