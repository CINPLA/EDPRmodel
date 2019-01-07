import pytest 
from pinsky_rinzel_pump.leakycell import *

def test_modules():
    test_cell = LeakyCell(279.3, 15., 145., 15., 145., 100., 3., 100., 3., 115., 148., 115., 148.)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721

#def test_I_diff():
#    D_k = [1.33e-9, 2.03e-9]
#    Z_k = [1, -1]
#    k_d = [14, 114]
#    k_s = [16, 116]
#    assert round(I_diff(3.2, D_k, Z_k, k_d, k_s), 3) == 1.319
#
#def test_sigma():
#    D_k = [1.33e-9, 2.03e-9]
#    Z_k = [1, -1]
#    k_d = [14, 114]
#    k_s = [16, 116]
#    assert round(sigma(3.2, D_k, Z_k, k_d, k_s), 3) == 0.099
