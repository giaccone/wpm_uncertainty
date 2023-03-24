# import modules
import numpy as np
import matplotlib.pyplot as plt
from exposure.modified.icnirp import origin_zero, real_zero, origin_pole, real_pole

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

# time domain vs frequency domain filtering
f = np.logspace(0, 5, 1000)
ft = 300
k = 2
phase1 = real_zero(ft, k, f)
phase2 = real_pole(ft, k, f)


fig1, axs = plt.subplots()
phase1_icnirp = np.zeros_like(phase1)
phase1_icnirp[f >= ft] = 90 * k
axs.semilogx(f, phase1_icnirp, 'C1', linewidth=3, label='frequency')
axs.semilogx(f, phase1, 'C2--', linewidth=3,label='frequency proposed')
axs.set_xticks((ft*0.1, ft, ft*10))
axs.set_xticklabels(("$0.1 f_j$", "$f_j$", "$10 f_j$"), fontsize=16)
axs.set_ylim(-200,200)
axs.set_yticks((-180, 0, 90, 180))
axs.set_yticklabels(("-90 $k$", "0", "45 $k$", "90 $k$"), fontsize=16)
axs.grid(linestyle='dashed')
axs.set_xlabel("frequency (Hz)", fontsize=16)
axs.set_ylabel("phase (deg)", fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()

fig2, axs = plt.subplots()
phase2_icnirp = np.zeros_like(phase2)
phase2_icnirp[f >= ft] = -90 * k
axs.semilogx(f, phase2_icnirp, 'C1', linewidth=3, label='frequency')
axs.semilogx(f, phase2, 'C2--', linewidth=3, label='frequency proposed')
axs.set_xticks((ft*0.1, ft, ft*10))
axs.set_xticklabels(("$0.1 f_j$", "$f_j$", "$10 f_j$"), fontsize=16)
axs.set_ylim(-200,200)
axs.set_yticks((-180, -90, 0, 180))
axs.set_yticklabels(("-90 $k$", "-45 $k$", "0", "90 $k$"), fontsize=16)
axs.grid(linestyle='dashed')
axs.set_xlabel("frequency (Hz)", fontsize=16)
axs.set_ylabel("phase (deg)", fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()

fig1.savefig("real_zero.png", dpi=300)
fig2.savefig("real_pole.png", dpi=300)

# show figures
plt.show()
