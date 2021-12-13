import numpy as np
import npquad
from scipy.special import erf


NthQuartStr = r"$t\sqrt{1 - (1 - \frac{ln(N)}{t})^{2}}$"


def theoreticalNthQuart(N, time):
    """
    Returns the predicted position of the 1/Nth quantile. Remember that the
    predicted position is twice the distance we're recording.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1

    time : numpy array
        Times to record the 1/Nth quantile for

    Returns
    -------
    theory : numpy array
        Theoretical 1/Nth quantile as a function of time predicted by the
        BC model for diffusion.
    """

    logN = np.log(N).astype(np.float64)
    theory = np.piecewise(
        time,
        [time < logN, time >= logN],
        [lambda x: x, lambda x: x * np.sqrt(1 - (1 - logN / x) ** 2)],
    )
    return theory


NthQuartVarStr = (
    r"$(2 ln(N))^{\frac{2}{3}} \frac{(t/ln(N) - 1)^\frac{4}{3}}{2t/ln(N) - 1}$"
)

def theoreticalVar(N, time, crossover=None, width=None):
    '''
    Returns the theoretical quantile variance over time. Does this by stitching
    together short time and long time with an error function.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    crossover : float (optional)
        Time when quantile variance shifts from short to long time regime

    width : float (optional)
        Error function width. A shorter width makes the switch from short to long
        time regimes more dramatic.

    Returns
    -------
    theory : numpy array
        Theoretical quantile variance at each time
    '''
    if crossover is None:
        crossover = np.log(N).astype(float) * 10**2
    if width is None:
        width = crossover

    theory_short = theoreticalNthQuartVar(N, time)
    theory_long = theoreticalNthQuartVarLargeTimes(N, time)
    error_func = (erf((time - crossover) / width) + 1 ) / 2
    theory = theory_short * (1-error_func) + theory_long * (error_func)
    return theory

def theoreticalNthQuartVar(N, time):
    """
    Returns the predicted position of the 1/Nth quantile variance over time.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    Returns
    -------
    theory : numpy array
        Theoretical 1/Nth quantile variance as a function of time
    """

    logN = np.log(N).astype(np.float64)
    return (2 * logN) ** (2 / 3) * (time / logN - 1) ** (4 / 3) / (2 * time / logN - 1)


NthQuartVarStrLargeTimes = r"$t^{1/2}\pi^{1/2}/2$"


def theoreticalNthQuartVarLargeTimes(N, time):
    """
    Returns the predicted position of the 1/Nth quantile variance over time for
    t ~ Log(N)^2.
    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    Returns
    -------
    theory : numpy array
        Theoretical 1/Nth quantile variance as a function of time
    """

    return time ** (1 / 2) * np.pi ** (1 / 2) / 2


def theoreticalPbatT(vs, t):
    """
    Get the theoretically predicted ln(Pb(vt, t)) (probability greater than
    index vt at current time)

    Parameters
    ----------
    t : int or float
        Time to calculate ln(Pb(vt, t)) at

    v : numpy array
        List of velocities

    Returns
    -------
    lnPbs : numpy array
        Natural log of probabilities greater than vt
    """
    M = -1.77  # Mean of TW distribution for beta=2
    I = 1 - np.sqrt(1 - vs ** 2)
    sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
    return -I * t + t ** (1 / 3) * sigma * M


PbMeanStr = r"$-I * t + t^{1/3}\sigma M$"


def theoreticalPbMean(v, t):
    """
    For a specified v get the probability of being greater than vt over time.
    Otherwise known as ln(Pb(vt,) t))

    Parameters
    ----------
    t : numpy float
        Times to calculate ln(Pb(vt, t)) for

    v : float
        Velocity to get probability of. Must satisfy 0 < v < 1.

    Returns
    -------
    numpy array
        Logged probability of being greater than vt or ln(Pb(vt, t)).
    """

    M = -1.77
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
    return -I * t + t ** (1 / 3) * sigma * M


PbVarStr = r"$t^{2/3}\sigma^{2} V$"


def theoreticalPbVar(v, t):
    """
    For a specified v get the variance of the probability of being greater than
    vt over time. Otherwise known as Var(ln(Pb(vt, t)))

    Parameters
    ----------
    t : numpy float
        Times to calculate ln(Pb(vt, t)) for

    v : float
        Velocity to get probability of. Must satisfy 0 < v < 1.

    Returns
    -------
    numpy array
        Variance of logged probability of being greater than vt or Var(ln(Pb(vt, t))).
    """

    V = 0.813
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
    theory = (t ** (2 / 3)) * sigma ** 2 * V
    return theory
