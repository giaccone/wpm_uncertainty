# import libraries
import numpy as np
from scipy.interpolate import interp1d


def csw(amplitude, phase, frequency, time):
    """

    Parameters
    ----------
    amplitude (float):
        amplitude of the continous sine wave
    phase (float):
        phase of the continous sine wave
    frequency (float):
        frequency of the continous sine wave
    time (np.array):
        array with time values

    Returns
    -------
    sine_wave (np.array):
        continous sine wave
    """
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return sine_wave


def sin_burst(amplitude, frequency,  n_period=9, shape=(0, 80, 90, 100, 100, 100, 90, 80, 0), spp=1001, delay=0):
    """

    Parameters
    ----------
    amplitude (float):
        peak of the sinusoidal burst
    frequency (float):
        fundamental frequency
    n_period (int):
        number of complete cycles in the sinusoidal burst
    shape (tuple):
        tuple including percentages of the amplitude. The length of the tuple is n_period
    spp (int):
        sample per period
    delay (float):
        delay in second

    Returns
    -------
    time (np.array):
        numpy array with time values
    wave (np.array):
        numpy array with the waveform

    """
    windows = n_period / frequency
    time = np.linspace(0, windows, n_period * spp)
    profile = (np.tile(np.array(shape).reshape(-1, 1), (1, spp))).flatten() / 100
    wave = profile * amplitude * np.sin(2 * np.pi * frequency * time)

    if delay == 0:
        pass
    else:
        ts = time[1] - time[0]
        time = time + delay
        t0 = np.sort(np.arange(delay - ts, 0, -ts))
        wave0 = np.zeros_like(t0)
        time = np.concatenate((t0, time))
        wave = np.concatenate((wave0, wave))
    return wave, time


def pulse(peak, tr, tau, tf, period, time, delay=None):
    """

    Parameters
    ----------
    peak (float):
        peak of the pulse
    tr (float):
        rise time
    tau (float):
        pulse duration
    tf (float):
        fall tile
    period (float):
        period of the waveform
    time (np.array):
        numpy array with times at which the pulse must be defined
    delay (float):
        delay before the pulse

    Returns
    -------
    wave (np.array):
        numpy array with the waveform

    """
    if delay is None:
        time_points = [0, tr, tr + tau, tr + tau + tf, period]
        wave_points = [0, peak, peak, 0, 0]
    else:
        time_points = [0, delay, tr + delay, tr + tau + delay, tr + tau + tf + delay, period]
        wave_points = [0, 0, peak, peak, 0, 0]

    # create linear interpolator
    fun = interp1d(time_points, wave_points, bounds_error=False,
                   fill_value=(wave_points[0], wave_points[-1]), assume_sorted=True)
    # interpolation
    wave = fun(time)

    return wave


def double_pulse(peak1, t1, tau1, t2, tau2, period, time):
    """

    Parameters
    ----------
    peak1 (float):
        peak of the first pulse
    t1 (float):
        begin of the first pulse
    tau1 (float):
        duration of the first pulse
    peak2 (float):
        peak of the second pulse
    t2 (float):
        begin of the second pulse
    tau2 (float):
        duration of the second pulse
    time (np.array):
        numpy array with times at which the pulse must be defined

    Returns
    -------
    wave (np.array):
        numpy array with the waveform

    """
    # peak2 is computed so that the mean value of the waveform is zero
    peak2 = -tau1 / tau2 * peak1

    dt = time[1] - time[0]
    if t1 == 0:
        time_points = [t1, tau1, tau1 + dt, t2-dt, t2, t2 + tau2, t2 + tau2 + dt, period]
        wave_points = [peak1, peak1, 0, 0, peak2, peak2, 0, 0]
    else:
        time_points = [0, t1-dt, t1, t1 + tau1, t1 + tau1 + dt, t2 - dt, t2, t2 + tau2, t2 + tau2 + dt, period]
        wave_points = [0, 0, peak1, peak1, 0, 0, peak2, peak2, 0, 0]

    # create linear interpolator
    fun = interp1d(time_points, wave_points, bounds_error=False,
                   fill_value=(wave_points[0], wave_points[-1]), assume_sorted=True)
    # interpolation
    wave = fun(time)

    return wave


def exp_pulse(peak, delay, taur, tauf, duration, npt):
    """

    Parameters
    ----------
    peak (float):
        peak of the pulse
    delay (float):
        delay before the pulse
    taur (float):
        time constant for the rise time.
        rise time duration is set to 7*taur
    tauf (float):
        time constant for the fall time.
        fall time duration is set to 7*taur
    duration (float):
        duration of the pulse
    npt (int):
        number of points defining the pulse

    Returns
    -------
    time (np.array):
        numpy array with time values
    wave (np.array):
        numpy array with the waveform

    """
    time = np.linspace(-delay, duration - delay, npt)
    wave = np.zeros_like(time)
    wave[(time >= 0) & (time < 7 * taur)] = peak * (1 - np.exp(-(time[(time >= 0) & (time < 7 * taur)])/taur))
    wave[time >= 7 * taur] = peak * (1 - np.exp(-7)) * np.exp(-(time[time >= 7*taur] - 7 * taur) / tauf)
    time = time - time[0]

    return time, wave


if __name__ == "__main__":

    # import modules
    import matplotlib.pyplot as plt

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

    # plot pulse
    time = np.linspace(0, 1, 1000)
    peak = 10
    tr = 0.05
    tf = 0.1
    tau = 0.3
    period = 1
    delay=0.1
    field = pulse(peak, tr, tau, tf, period, time, delay=delay)

    hp = plt.figure()
    plt.plot(time, field)
    plt.xticks([0, delay, delay + tr, delay + tr + tau, delay + tr + tau + tf, period],
               ['', '$t_D$', '', '', '', ''], fontsize=18)
    plt.yticks([0, peak], ['', '$A$'], fontsize=18)
    plt.grid(linestyle='dashed')
    plt.xlim([0, period])
    plt.ylim([-3.5, 10.5])
    plt.xlabel('time', fontsize=18)

    # plt.annotate(text='', xy=(0, -1.5), xytext=(delay,-1.5), arrowprops=dict(arrowstyle='<->'))
    # plt.text(delay/3 , -1.1, "$t_D$", fontsize=18)
    plt.annotate(text='', xy=(delay, -1.5), xytext=(delay + tr,-1.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + tr/4, -1.1, "$t_R$", fontsize=18)
    plt.annotate(text='', xy=(delay + tr, -1.5), xytext=(delay + tr + tau,-1.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + tr + tau/3, -1.1, "$\\tau$", fontsize=18)
    plt.annotate(text='', xy=(delay + tr + tau, -1.5), xytext=(delay + tr + tau + tf,-1.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + tr + tau + tf/3, -1.1, "$t_F$", fontsize=18)
    plt.annotate(text='', xy=(0, -3), xytext=(period,-3), arrowprops=dict(arrowstyle='<->'))
    plt.text(period * 0.8, -2.6, "$T$", fontsize=18)
    plt.tight_layout()

    # plot sine
    amplitude = 10
    frequency = 1
    delay = 1
    field, time = sin_burst(amplitude, frequency, delay=delay, n_period=11, shape=(0, 0, 80, 90, 100, 100, 100, 90, 80, 0, 0))

    hs = plt.figure()
    plt.plot(time, field)
    plt.xticks([0, delay, delay + 2/frequency, delay + 3/frequency, delay + 4/frequency, delay + 7/frequency, delay + 8/frequency,
                delay + 9/frequency, delay + 11/frequency],
                ['0','$t_D$','','','','','','',''], fontsize=18)
    plt.yticks([-amplitude, 0,  amplitude], ['$-A$', '0', '$A$'], fontsize=18)
    plt.grid(linestyle='dashed')
    # plt.xlim([0, period])
    plt.ylim([-15, 11])
    plt.xlabel('time', fontsize=18)
    # plt.annotate(text='', xy=(0, -13), xytext=(delay,-13), arrowprops=dict(arrowstyle='<->'))
    # plt.text(delay/3, -12.5, '$d$', fontsize=18)
    plt.annotate(text='', xy=(delay, -13), xytext=(delay + 2/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 0.4, -12.0,"$N'_{0}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 2/frequency, -13), xytext=(delay + 3/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 2/frequency + 0.15, -12.0,"$N'_{80}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 3/frequency, -13), xytext=(delay + 4/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 3/frequency + 0.15, -12.0,"$N'_{90}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 4/frequency, -13), xytext=(delay + 7/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 5/frequency + 0.15, -12.0,"$N_{100}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 7/frequency, -13), xytext=(delay + 8/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 7/frequency + 0.15, -12.0,"$N''_{90}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 8/frequency, -13), xytext=(delay + 9/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 8/frequency + 0.15, -12.0,"$N''_{80}$", fontsize=16)
    plt.annotate(text='', xy=(delay + 9/frequency, -13), xytext=(delay + 11/frequency,-13), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 9.5/frequency + 0.15, -12.0,"$N''_{0}$", fontsize=16)
    plt.tight_layout()


    # plot double pulse
    time = np.linspace(0, 10, 1000)
    peak1 = 10
    t1 = 1 
    tau1 = 1
    t2 = 5
    tau2 = 3
    period = 10
    hdp = plt.figure()

    field = double_pulse(peak1, t1, tau1, t2, tau2, period, time)

    plt.plot(time, field)
    plt.xticks([0, t1, t1+tau1, t2, t2+tau2, period], ['','$t_1$', '', '$t_2$', '', ''], fontsize=18)
    plt.yticks([-tau1/tau2 * peak1, 0, peak1], ['$A_2$', '0', '$A_1$'], fontsize=18)
    plt.ylim(-8, 12)
    plt.xlabel('time', fontsize=18)
    plt.annotate(text='', xy=(t1, -5.5), xytext=(t1+tau1,-5.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(t1 + tau1/3, -5, '$\\tau_1$', fontsize=18)
    plt.annotate(text='', xy=(t2, -5.5), xytext=(t2+tau2,-5.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(t2 + tau2/3, -5, '$\\tau_2$', fontsize=18)
    plt.annotate(text='', xy=(0, -7), xytext=(period,-7), arrowprops=dict(arrowstyle='<->'))
    plt.text(period/3, -6.5, '$T$', fontsize=18)
    plt.grid(linestyle='dashed')
    plt.tight_layout()

    # plot exp-pulse
    peak = 10
    delay = 1
    taur = 0.2
    tauf = 0.4
    duration = 10
    npt = 1001
    time, field = exp_pulse(peak, delay, taur, tauf, duration, npt)

    he = plt.figure()
    plt.plot(time, field)
    plt.xticks([0, delay, delay + 7 * taur, delay + 7*taur + 7*tauf, duration],
                ['0', '$t_D$', '', '', ''], fontsize=18)
    plt.yticks([0, peak], ['0', '$A$'], fontsize=18)
    plt.annotate(text='', xy=(delay, -2), xytext=(delay + 7*taur,-2), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 7*taur*0.3, -1.5, '$7 \\tau_r$', fontsize=18)
    plt.annotate(text='', xy=(delay + 7*taur, -2), xytext=(delay + 7*taur + 7*tauf,-2), arrowprops=dict(arrowstyle='<->'))
    plt.text(delay + 7*taur + 7*tauf*0.3, -1.5, '$7 \\tau_f$', fontsize=18)
    plt.annotate(text='', xy=(0, -3), xytext=(duration,-3), arrowprops=dict(arrowstyle='<->'))
    plt.text(duration*0.6, -2.5, '$T$', fontsize=18)

    plt.ylim(-4, 11)
    plt.xlabel('time', fontsize=18)
    plt.grid()
    plt.tight_layout()

    # # save figures
    # hp.savefig('wave_pulse', dpi=300)
    # hs.savefig('wave_sine', dpi=300)
    # hdp.savefig('wave_double_pulse', dpi=300)
    # he.savefig('wave_exp', dpi=300)

    # show figures
    plt.show()
