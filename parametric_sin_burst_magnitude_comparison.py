# load modules
from exposure.icnirp import icnirp_filter
from scipy.signal import bode
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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


# Compute ration between frequency and time domain
# all possible parameters
year = '2010'
receptor = 'occupational'
quantity = 'B'

# frequency
domain = 'freq'
f1 = np.logspace(1, 5, 75)
H_f, theta_f = icnirp_filter(year, receptor, quantity, domain, f1)

# time
domain = 'time'
num, den = icnirp_filter(year, receptor, quantity, domain)
omega, Hdb, theta_t = bode((num, den), 2 * np.pi * f1)
H_t = 10 ** (Hdb / 20)


# load parametric analysis for sin-burst
data = np.load('./parametric_data/sin_burst.npz')
f = data['frequency']
mu = data['mu']
sigma = data['sigma']


hf = plt.figure()
# plt.semilogx(f, mu[:, 0], label='time')
# plt.fill_between(f, mu[:, 0] - 2*sigma[:, 0], mu[:, 0] + 2*sigma[:, 0], color="C0", alpha=0.2)
plt.semilogx(f, mu[:, 1] / mu[:, 0], 'C1', label='frequency/time')
# plt.fill_between(f, mu[:, 1] - 2*sigma[:, 1], mu[:, 1] + 2*sigma[:, 1], color="C1", alpha=0.2)
plt.semilogx(f, mu[:, 2] / mu[:, 0], 'C2', label='frequency proposed / time')
# plt.fill_between(f, mu[:, 2] - 2*sigma[:, 2], mu[:, 2] + 2*sigma[:, 2], color="C2", alpha=0.2)
plt.semilogx(f1, (H_f / H_t), 'ko', mfc='none', label='ratio $WF_\mathrm{f}/WF_\mathrm{t}$')

lg = plt.legend(fontsize=13)
lg.set_draggable(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('frequency (Hz)', fontsize=16)
plt.ylabel('exposure index', fontsize=16)
plt.tight_layout()
plt.show()


