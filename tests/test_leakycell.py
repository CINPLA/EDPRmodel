import pytest 
from pinsky_rinzel_pump.leakycell import *

def test_modules():

    test_cell = LeakyCell(279.3, 14., 145., 16., 145., 100., 3., 100., 3., 115., 148., 115., 148.)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.Na_si, test_cell.Na_di), 4) == 0.0078

    assert test_cell.total_charge( \
        [test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], 1e-1) == -9648
    
    assert test_cell.total_charge( \
        [10, 10, 20], 10) == 0 

