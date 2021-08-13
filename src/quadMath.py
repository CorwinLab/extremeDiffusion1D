import numpy as np
import npquad


def logspace(start, stop, num=50, endpoint=False):
    """
    Returns numbers evenly spaced on a log scale.

    Parameters
    ----------
    start : int
        Starting exponent value of the sequence. Start is 10 ** start

    stop : int
        Ending exponent value of the sequence. Stop is 10 ** stop

    num : int (50)
        Number of samples to generate

    endpoint : bool (False)
        If true, stop is the last sample. Otherwise not included.

    Returns
    -------
    samples : numpy array (dtype = np.quad)
        num samples, evenly spaced on a log scale.
    """

    if stop > 4900:
        raise ValueError(f"Stop exponent cannot be > 4900, but is {stop}")

    samples = [
        np.quad(f"1e{i}") for i in np.linspace(start, stop, num, endpoint=endpoint)
    ]
    return np.array(samples, dtype=np.quad)


def logarange(start, stop, step_size=1, endpoint=False):
    """
    Returns numbers space on log scale with specified step size. Similiar to
    logspace but uses a step size.

    Parameters
    ----------
    start : int
        Starting exponent value of the sequence. Start is 10 ** start

    stop : int
        Ending exponent value of the sequence. Stop is 10 ** stop

    step_size : int (1)
        Exponent step size between samples

    endpoint : bool (False)
        If true, stop is the last sample. Otherwise not included.

    Returns
    -------
    samples : numpy array (dtype = np.quad)
        num samples, evenly spaced on a log scale.
    """

    if stop > 4900:
        raise ValueError(f"Stop exponent cannot be > 4900, but is {stop}")

    if endpoint:
        stop += step_size

    samples = [np.quad(f"1e{i}") for i in np.arange(start, stop, step_size)]
    return np.array(samples, dtype=np.quad)
