# import modules
import numpy as np
from scipy.signal import bode
from exposure.icnirp import icnirp_limit, icnirp_filter
from exposure.modified.icnirp import icnirp_filter as icnirp_filter_beta
from exposure.pulsed import wpm_time, wpm_freq
import matplotlib.pyplot as plt
import fourier.ffttools as ft
from waveform import sin_burst, pulse, double_pulse, exp_pulse

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

# define filter
year = '2010'
receptor = 'occupational'
quantity = 'Ecns'

# select waveform
wave = 'double_pulse'
if wave == 'sin_burst':
    fr = 400
    Glim = icnirp_limit(fr, year, receptor, quantity) * np.sqrt(2)
    field, time = sin_burst(Glim, fr, n_period=11, shape=(0, 0, 80, 90, 100, 100, 100, 90, 80, 0, 0))
elif wave == 'pulse':
    delay = 100e-3
    tr = 5e-3
    tau = 300e-3
    tf = 10e-3
    period = 1000e-3
    time = np.linspace(0, period, 10001)
    peak = icnirp_limit(1 / tr, year, receptor, quantity) * np.sqrt(2)
    field = pulse(peak, tr, tau, tf, period, time, delay)
elif wave == 'double_pulse':
    t1 = 100e-3
    tau1 = 100e-3
    t2 = 300e-3
    peak1 = icnirp_limit(1 / tau1, year, receptor, quantity) * np.sqrt(2)
    # peak2 = -peak1 * 0.8
    tau2 = 200e-3
    period = 1000e-3
    time = np.linspace(0, period, 10001)
    field = double_pulse(peak1, t1, tau1, t2, tau2, period, time)
elif wave == 'exp_pulse':
    peak = 1000e-6
    delay = 100e-3
    taur = 20e-3
    tauf = 50e-3
    duration = 1000e-3
    npt = 10001
    time, field = exp_pulse(peak, delay, taur, tauf, duration, npt)

# Weighted peak method ...
# ... in time domain
num, den = icnirp_filter(year, receptor, quantity, 'time')
EIt, Wt = wpm_time(num, den, time, field, makeplot='n')

# ... in frequency domain
freq, F = ft.fft_analysis(time, field)
weight_fun, phase = icnirp_filter(year, receptor, quantity, 'freq', f=freq)
EIf, Wf = wpm_freq(weight_fun, phase, freq, F, makeplot='n')

# ... in frequency domain (proposed)
weight_fun, phase = icnirp_filter_beta(year, receptor, quantity, 'freq', f=freq)
EIfb, Wfb = wpm_freq(weight_fun, phase, freq, F, makeplot='n')

k = 1 / EIt
EIt, EIf, EIfb = [ele * k for ele in [EIt, EIf, EIfb]]
Wt, Wf, Wfb = [ele * k for ele in [Wt, Wf, Wfb]]

print("Exposure index")
print(f"    * time\t\t\t{EIt :.2f}")
print(f"    * frequency\t\t\t{EIf :.2f}")
print(f"    * frequency proposed\t{EIfb :.2f}")

# plot field
h1 = plt.figure()
plt.plot(time * 1e3, field * k * 1e6)
plt.xlabel('time (ms)', fontsize=16)
plt.ylabel('field (uT)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# plot weighted field
h2 = plt.figure()
plt.plot(time, np.abs(Wt), label=f'time, EI={EIt :.2f}')
plt.plot(time, np.abs(Wf), label=f'frequency, EI={EIf :.2f}')
plt.plot(time, np.abs(Wfb), label=f'frequency proposed, EI={EIfb :.2f}')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
lg = plt.legend(fontsize=14)
lg.set_draggable(True)
plt.xlabel('time (ms)', fontsize=16)
plt.ylabel('weighted field', fontsize=16)
plt.tight_layout()

# time domain vs frequency domain filtering
frequency = np.logspace(0, 5, 250)
# time
H_f, theta_f = icnirp_filter(year, receptor, quantity, 'freq', frequency)
H_fb, theta_fb = icnirp_filter_beta(year, receptor, quantity, 'freq', frequency)
# frequency
num, den = icnirp_filter(year, receptor, quantity, 'time')
omega, Hdb, theta_t = bode((num, den), 2 * np.pi * frequency)
H_t = 10 ** (Hdb / 20)

# filter comparison
hf = plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(frequency, H_t)
plt.loglog(frequency, H_f)
plt.loglog(frequency, H_fb, linestyle='--')
plt.ylabel('magnitude $\mathrm{(T)}^{-1}$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.subplot(2, 1, 2)
plt.semilogx(frequency, theta_t)
plt.semilogx(frequency, theta_f)
plt.semilogx(frequency, theta_fb, linestyle='--')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('phase (deg)', fontsize=14)
plt.xlabel('frequency (Hz)', fontsize=14)
lg = plt.legend(('time', 'frequency', 'frequency proposed'), fontsize=14)
lg.set_draggable(True)
plt.tight_layout()

# show figures
plt.show()

#h2.savefig(wave + "_weighted", dpi=300)
#hf.savefig("B_filter", dpi=300)


