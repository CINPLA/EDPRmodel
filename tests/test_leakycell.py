import pytest
import numpy as np
from pinsky_rinzel_pump.leakycell import *

def dkdt(t,k):

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = k

    my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de)

    dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de = my_cell.dkdt()

    return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de

def test_modules():

    test_cell = LeakyCell(279.3, 14., 145., 16., 145., 100., 3., 100., 3., 115., 148., 115., 148.)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.Na_si, test_cell.Na_di), 4) == 0.0078

    assert test_cell.total_charge( \
        [test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], 1e-1) == -9648
    
    assert test_cell.total_charge( \
        [10, 10, 20], 10) == 0 

def test_charge_conservation():
    """Tests that no charge disappear."""
    
    EPS = 1e-14

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 115.
    Cl_se0 = 148.
    Na_di0 = 15.
    Na_de0 = 145.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 115.
    Cl_de0 = 148.

    t_span = (0, 100)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    t = sol.t
    
    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de)

    Na_tot_charge = np.round(test_cell.D_Na*(Na_si*test_cell.V_si + Na_se*test_cell.V_se + Na_di*test_cell.V_di + Na_de*test_cell.V_de) * test_cell.F)
    K_tot_charge = np.round(test_cell.D_K*(K_si*test_cell.V_si + K_se*test_cell.V_se + K_di*test_cell.V_di + K_de*test_cell.V_de) * test_cell.F)
    Cl_tot_charge = np.round(test_cell.D_Cl*(Cl_si*test_cell.V_si + Cl_se*test_cell.V_se + Cl_di*test_cell.V_di + Cl_de*test_cell.V_de) * test_cell.F)
     
    Na_diff_charge = np.diff(Na_tot_charge)
    K_diff_charge = np.diff(K_tot_charge)
    Cl_diff_charge = np.diff(Cl_tot_charge)

    N = len(Na_diff_charge)
    for i in range(N):
        assert Na_diff_charge[i] < EPS
        assert K_diff_charge[i] < EPS
        assert Cl_diff_charge[i] < EPS

def test_charge_symmetry():
    """Tests that Q_i = -Q_e."""

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 115.
    Cl_se0 = 148.
    Na_di0 = 15.
    Na_de0 = 145.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 115.
    Cl_de0 = 148.

    t_span = (0, 100)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    t = sol.t
    
    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de)

    q_si = test_cell.total_charge([test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], \
        test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.Na_se, test_cell.K_se, test_cell.Cl_se], \
        test_cell.V_se)
    
    q_di = test_cell.total_charge([test_cell.Na_di, test_cell.K_di, test_cell.Cl_di], \
        test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.Na_de, test_cell.K_de, test_cell.Cl_de], \
        test_cell.V_de)

    N = len(q_si)
    for i in range(N):
        assert round(q_si[i], 5) == -round(q_se[i], 5)
        assert round(q_di[i], 5) == -round(q_de[i], 5)

def test_charge_conservation_w_diffusion():
    """Tests that no charge disappear when ion diffuse."""
    
    EPS = 1e-14

    Na_si0 = 12.
    Na_se0 = 142.
    K_si0 = 99.
    K_se0 = 2.
    Cl_si0 = 111.
    Cl_se0 = 144.
    Na_di0 = 18.
    Na_de0 = 148.
    K_di0 = 101.
    K_de0 = 4.
    Cl_di0 = 119.
    Cl_de0 = 152.

    t_span = (0, 10)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    t = sol.t
    
    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de)

    Na_tot_charge = np.round(test_cell.D_Na*(Na_si*test_cell.V_si + Na_se*test_cell.V_se + Na_di*test_cell.V_di + Na_de*test_cell.V_de) * test_cell.F)
    K_tot_charge = np.round(test_cell.D_K*(K_si*test_cell.V_si + K_se*test_cell.V_se + K_di*test_cell.V_di + K_de*test_cell.V_de) * test_cell.F)
    Cl_tot_charge = np.round(test_cell.D_Cl*(Cl_si*test_cell.V_si + Cl_se*test_cell.V_se + Cl_di*test_cell.V_di + Cl_de*test_cell.V_de) * test_cell.F)
     
    Na_diff_charge = np.diff(Na_tot_charge)
    K_diff_charge = np.diff(K_tot_charge)
    Cl_diff_charge = np.diff(Cl_tot_charge)

    N = len(Na_diff_charge)
    for i in range(N):
        assert Na_diff_charge[i] < EPS
        assert K_diff_charge[i] < EPS
        assert Cl_diff_charge[i] < EPS
