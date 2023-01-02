
import numpy as np
import matplotlib.pyplot as plt
import fourier.ffttools as ft

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

# set delays
rise_times = ['0', '10', '20', '30']

# begin cycle
for k, tr in enumerate(rise_times):
    # set desired weld time value
    wt = '200'

    # load data
    try:
        fname = './currents/current_delay_' + tr + '_wt_' + wt + '.npz'
        data = np.load(fname)
    except FileNotFoundError:
        msg = f"File {fname} not found.\n\nYou likely need to download the data\n"
        msg += f"Instruction for download in 'currents.txt' in the currents folder."
        raise FileNotFoundError(msg)
    time = data['time']
    current = data['current']

    # initialize/update figure
    if k == 0:
        h1 = plt.figure()
    else:
        h1 = plt.figure(h1.number)
    
    # add trace
    label = '$t_R={}$ ms'.format(int(tr))
    plt.plot(time*1e3, current*1e-3, label=label)

    # Compute spectrum
    f, X, fp, Xp = ft.fft_analysis(time, current, pos='y')
    Xp[np.abs(Xp) < np.max(np.abs(Xp)*0.0015)] = 0

    # initialize/update figure
    if k == 0:
        h2 = plt.figure()
    else:
        h2 = plt.figure(h2.number)

    # add trace
    plt.loglog(fp, np.abs(Xp)*1e-3, '--o', label=label, ms=4)

# finalize figure
plt.figure(h1.number)
plt.xlabel('time (ms)', fontsize=16)
plt.ylabel('current (kA)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.xlim(time[0]*1e3, time[-1]*1e3)
plt.tight_layout()

plt.figure(h2.number)
plt.xlim(1, 10000)
plt.xticks([1, 10, 100, 300, 2000, 4000], ['1', '10', '100', '300','2k', '4k'], fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(1e-2, 15)
plt.legend(fontsize=14)
plt.xlabel('frequency (Hz)', fontsize=16)
plt.ylabel('current (kA)', fontsize=16)
plt.tight_layout()
plt.show()

# h1.savefig('current_wt200_variable_tr', dpi=300)
# h2.savefig('spectrum_wt200_variable_tr', dpi=300)

plt.show()