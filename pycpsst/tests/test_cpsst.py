import numpy as np

from pycpsst import ChangePointSST


def test_output_shape():
    '''
    Check whether the output of the
    ChangePointSST().score(X) method
    is a float.
    '''

    signal = np.random.randn(200)
    cp = ChangePointSST()

    result = cp.score(signal)

    assert isinstance(result, float)
