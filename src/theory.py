import numpy as np
import npquad


def theoreticalNthQuart(N, time):
    """
    Returns the predicted position of the 1/Nth quartile. Remember that the
    predicted position is twice the distance we're recording.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quartile to measure. Should be > 1

    time : numpy array
        Times to record the 1/Nth quartile for

    Returns
    -------
    theory : numpy array
        Theoretical 1/Nth quartile as a function of time predicted by the
        BC model for diffusion.
    """

    logN = np.log(N).astype(np.float64)
    theory = np.piecewise(
        time,
        [time < logN, time >= logN],
        [lambda x: x, lambda x: x * np.sqrt(1 - (1 - logN / x) ** 2)],
    )
    return theory


def theoreticalNthQuartVar(N, time):
    """
    Returns the predicted position of the 1/Nth quartile variance over time.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quartile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quartile variance for.

    Returns
    -------
    theory : numpy array
        Theoretical 1/Nth quartile variance as a function of time
    """

    logN = np.log(N).astype(np.float64)
    return (2 * logN) ** (2 / 3) * (time / logN - 1) ** (4 / 3) / (2 * time / logN - 1)


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
    theory = (time ** (2 / 3)) * sigma ** 2 * V
    return theory
