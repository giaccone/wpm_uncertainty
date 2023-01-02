import numpy as np
from exposure.icnirp import icnirp_filter
from exposure.modified.icnirp import icnirp_filter as icnirp_filter_proposed
from exposure.pulsed import wpm_time, wpm_freq
import fourier.ffttools as ft

# set rise times and weld times
rise_times = ['0', '10', '20', '30']
wts = ['75', '100', '125', '150', '175', '200']

# Weighted peak method parameters
year = '2010'
receptor = 'occupational'
quantity = 'B'

# begin cycle
for tr in rise_times:
    print(f"rise time: {tr} ms:")

    for wt in wts:
        # load data (field is scaled to 2500 uT)
        try:
            fname = './currents/current_delay_' + tr + '_wt_' + wt + '.npz'
            data = np.load(fname)
        except FileNotFoundError:
            msg = f"File {fname} not found.\n\nYou likely need to download the data\n"
            msg += f"Instruction for download in 'currents.txt' in the currents folder."
            raise FileNotFoundError(msg)


        time = data['time']
        field = data['current'] / np.max(data['current']) * 2500e-6

        # ... in time domain
        num, den = icnirp_filter(year, receptor, quantity, 'time')
        EIt, Wt  = wpm_time(num, den, time, field, makeplot='n')

        # ... in frequency domain
        freq, F = ft.fft_analysis(time, field)
        weight_fun, phase = icnirp_filter(year, receptor, quantity, 'freq', f=freq)
        EIf, Wf = wpm_freq(weight_fun, phase, freq, F, makeplot='n')

        # ... in frequency domain (proposed)
        weight_fun, phase = icnirp_filter_proposed(year, receptor, quantity, 'freq', f=freq)
        EIfb, Wfb = wpm_freq(weight_fun, phase, freq, F, makeplot='n')

        k = 1 / EIt
        EItn, EIfn, EIfbn = [ele * k for ele in [EIt, EIf, EIfb]]

        print("{}ms\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(wt, EIt, EIf, EIfb, EItn, EIfn, EIfbn))
        if wt == '200':
            print('-----------------------------')


