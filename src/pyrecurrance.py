import sys

sys.path.append("../recuranceRelation")
import recurrance as rec
import numpy as np
import npquad


class Recurrsion(rec.Recurrance):
    """
    Implementation of the recurrance relation from Ivan's write up.

    Parameters
    ----------
    beta : float
        Value of beta for beta distribution to draw from

    zB : numpy array
        Numpy array of size (tMax, tMax)

    tMax : int
        Maximum time to go out to and the size of the zB array

    Examples
    --------
    >>> rec = Recurrsion(beta=1, tMax=10)
    >>> rec.makeRec()
    >>> print(rec.zB.shape)
    (10, 10)
    >>> qs = rec.findQuintiles([10, 100])
    >>> print(qs)
    """

    def __str__(self):
        return f"Recurrsion(beta={self.beta}, tMax={self.tMax})"

    def __repr__(self):
        return self.__str__()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def zB(self):
        return np.array(self.getzB(), dtype=np.quad)

    @property
    def tMax(self):
        return self.gettMax()

    def makeRec(self):
        """
        Run the recurrance relation to populate the zB array. All elements on the
        diagonal will be initialized to 1.
        """

        super().makeRec()

    def findQuintile(self, N):
        """
        Find the corresponding quintile according to Ivan's predicted relationship.

        Parameters
        ----------
        N : int
            Quintile to measure over time

        Returns
        -------
        numpy array
            Quintile position over time
        """

        return np.array(super().findQuintile(N))

    def findQuintiles(self, Ns):
        """
        Find multiple quintiles according to Ivan's predicted relationship.

        Parameters
        ----------
        Ns : list or numpy array
            Quintiles to measure over time

        Returns
        -------
        qs : numpy array
            Quantiles position over time. Columns are the different quintiles and
            rows are different times.

        Examples
        --------
        >>> quintiles = [5, 10, 100]
        >>> rec = Recurrsion(np.inf, 1000)
        >>> rec.makeRec()
        >>> qc = rec.findQuintiles(quintiles)
        >>> print(qc)
        """

        # Need to do a bit of np magic to get it back into the proper form
        return np.fliplr(np.array(super().findQuintiles(Ns)).T)
