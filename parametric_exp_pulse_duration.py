# import common modules
import numpy as np
from progress.bar import Bar
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# exposure modules
from exposure.icnirp import icnirp_filter
from exposure.modified.icnirp import icnirp_filter as icnirp_filter_proposed
from exposure.pulsed import wpm_time, wpm_freq
import fourier.ffttools as ft
# polynomial chaos expansion
from pce import pce
# waveform
from waveform import exp_pulse

# optional: set LaTeX fonts
latex = True

if latex:
    # set LaTeX font
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    # for Palatino and other serif fonts use:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })


# define frequency range
tauf = np.linspace(20, 100, 500) * 1e-3

# define filter
year = '2010'
receptor = 'occupational'
quantity = 'B'

# common parameter
delay = 10e-3
duration = 1000e-3
taur = 10e-3
peak = 1000e-6
npt = 10001

# PCE Analysis
mu = []
sigma = []
t1 = timer()
print('Polynomial Chaos Expansion (PCE):')
with Bar('    * progress: ', max=len(tauf), suffix='%(percent)d%%') as bar:
    for f in tauf:

        # Wapper to main function
        def fun_time(x, f=f, y=year, r=receptor, q=quantity):
            def inner_fun(xx):
                # pulse
                time, field = exp_pulse(peak, delay, taur, f * xx[0], duration, npt)

                # time domain
                num, den = icnirp_filter(y, r, q, 'time')
                EI, _ = wpm_time(num, den, time, field, makeplot='n')

                return EI

            from joblib import Parallel, delayed
            y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))
            return np.array(y)

        # Wapper to main function
        def fun_freq(x, f=f, y=year, r=receptor, q=quantity):
            def inner_fun(xx):
                # pulse
                time, field = exp_pulse(peak, delay, taur, f * xx[0], duration, npt)

                # frequency domain
                freq, F = ft.fft_analysis(time, field)
                weight_fun, phase = icnirp_filter(y, r, q, 'freq', f=freq)
                EI, _ = wpm_freq(weight_fun, phase, freq, F)

                return EI

            from joblib import Parallel, delayed
            y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))
            return np.array(y)

        # Wapper to main function
        def fun_freq_beta(x, f=f, y=year, r=receptor, q=quantity):
            def inner_fun(xx):
                time, field = exp_pulse(peak, delay, taur, f * xx[0], duration, npt)

                # frequency domain
                freq, F = ft.fft_analysis(time, field)
                weight_fun, phase = icnirp_filter_proposed(y, r, q, 'freq', f=freq)
                EI, _ = wpm_freq(weight_fun, phase, freq, F)

                return EI

            from joblib import Parallel, delayed
            y = Parallel(n_jobs=20, verbose=0)(map(delayed(inner_fun), x))
            return np.array(y)


        # generate PCE
        order = 12
        distrib = ['n']
        param = [[1, 0.05]]
        level = 7

        poly_time = pce.PolyChaos(order, distrib, param)
        poly_time.spectral_projection(fun_time, level, verbose='n')
        poly_time.norm_fit()

        poly_freq = pce.PolyChaos(order, distrib, param)
        poly_freq.spectral_projection(fun_freq, level, verbose='n')
        poly_freq.norm_fit()

        poly_freq_proposed = pce.PolyChaos(order, distrib, param)
        poly_freq_proposed.spectral_projection(fun_freq_beta, level, verbose='n')
        poly_freq_proposed.norm_fit()

        mu.append([poly_time.mu, poly_freq.mu, poly_freq_proposed.mu])
        sigma.append([poly_time.sigma, poly_freq.sigma, poly_freq_proposed.sigma])
        bar.next()
t2 = timer()
print("    * elapsed time {:.3f} sec\n".format(t2 - t1))

# Normalize to 1 for time domain
mu = np.array(mu)
sigma = np.array(sigma)
# scale_factor = np.reshape(1 / mu[:, 0], (-1, 1))
scale_factor = 1 / mu[0, 0]
mu = mu * scale_factor
sigma = sigma * scale_factor
ci_factor = 1.96

# plot results
hf = plt.figure()
plt.plot(tauf * 1e3, mu[:, 0], label='time')
plt.fill_between(tauf * 1e3, mu[:, 0] - ci_factor * sigma[:, 0], mu[:, 0] + ci_factor * sigma[:, 0], color="C0", alpha=0.2)
plt.plot(tauf * 1e3, mu[:, 1], label='frequency')
plt.fill_between(tauf * 1e3, mu[:, 1] - ci_factor * sigma[:, 1], mu[:, 1] + ci_factor * sigma[:, 1], color="C1", alpha=0.2)
plt.plot(tauf * 1e3, mu[:, 2], label='frequency proposed')
plt.fill_between(tauf * 1e3, mu[:, 2] - ci_factor * sigma[:, 2], mu[:, 2] + ci_factor * sigma[:, 2], color="C2", alpha=0.2)
lg = plt.legend(fontsize=14)
lg.set_draggable(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\\tau_F$ (ms)', fontsize=16)
plt.ylabel('exposure index', fontsize=16)
plt.tight_layout()
plt.show()

# average/max mean values
mu_ave = np.average(mu, axis=0)
mu_max = np.max(mu, axis=0)

# average/max standard deviation
sigma_ave = np.average(sigma, axis=0)
sigma_max = np.max(sigma, axis=0)

# confidence interval
ci_ave = 2 * ci_factor * np.average(sigma / mu, axis=0) * 100
ci_max = 2 * ci_factor * np.max(sigma / mu, axis=0) * 100

print("mu ave: {:.3f} & {:.3f} & {:.3f}\\\\".format(*mu_ave))
print("mu max: {:.3f} & {:.3f} & {:.3f}\\\\\n".format(*mu_max))

print("sigma ave: {:.3e} & {:.3e} & {:.3e}\\\\".format(*sigma_ave))
print("sigma max: {:.3e} & {:.3e} & {:.3e}\\\\\n".format(*sigma_max))

print("ci ave: {:.2f} & {:.2f} & {:.2f}\\\\".format(*ci_ave))
print("ci max: {:.2f} & {:.2f} & {:.2f}\\\\\n".format(*ci_max))

np.savez_compressed('./parametric_data/exp_pulse', taur=taur,
                    mu=mu, sigma=sigma)
