from nose.tools import assert_equal
from nose.tools import assert_raises
import numpy as np

from jacobi import update_stream_function
from cfd import cfd
      
def test_empty_psi():
    psi=[]
    update_stream_function(0, 0, psi)
    assert_equal([], psi, 2)

def test_zero_psi():
    psi = [[0 for col in range(3)] for row in range(3)]
    expected = [[0 for col in range(3)] for row in range(3)]
    update_stream_function(1, 1, psi)
    np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

def test_ones_psi():
    psi = [[1 for col in range(5)] for row in range(5)]
    expected = [[1 for col in range(5)] for row in range(5)]
    update_stream_function(3, 3, psi)
    np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

def test_psi():
    psi = [[0 for col in range(5)] for row in range(5)]
    expected = [[0 for col in range(5)] for row in range(5)]
    # Before:
    # 0 0 1 0 0
    # 0 0 1 0 0
    # 0 0 1 1 1
    # 0 0 0 0 0
    # 0 0 0 0 0
    psi[0][2] = 1
    psi[1][2] = 1
    psi[2][2] = 1
    psi[2][3] = 1
    psi[2][4] = 1
    # Expect:
    # 0 0    1    0    0
    # 0 0.25 0.5  0.5  0
    # 0 0.25 0.5  0.5  1
    # 0 0    0.25 0.25 0
    # 0 0    0    0    0
    expected[0][2] = 1
    expected[1][1] = 0.25
    expected[1][2] = 0.5
    expected[1][3] = 0.5
    expected[2][1] = 0.25
    expected[2][2] = 0.5
    expected[2][3] = 0.5
    expected[2][4] = 1
    expected[3][2] = 0.25
    expected[3][3] = 0.25
    update_stream_function(3, 3, psi)
    np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

def test_cfd_invalid_inlet_width():
    assert_raises(AssertionError, cfd, 1000, 32, 20, 20, 5, 15, "out.dat")  
