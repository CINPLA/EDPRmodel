import pytest
from pinsky_rinzel_pump.leakycell import *

def test_nernst_potential():
    test_cell = LeakyCell(279.3, 15., 145., 15., 145., 100., 3., 100., 3., 115., 148., 115., 148.)
    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721
