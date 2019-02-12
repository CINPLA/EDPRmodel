import pytest
import numpy as np
from pinsky_rinzel_pump.leakycell import *

def dkdt(t,k):

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = k

    my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de,  0)

    dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de = my_cell.dkdt()

    return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de

def test_modules():

    test_cell = LeakyCell(279.3, 14., 145., 16., 145., 100., 3., 100., 3., 115., 148., 115., 148., 0, 0, 0, 0, 0)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.Na_si, test_cell.Na_di), 4) == 0.0078

    assert test_cell.total_charge( \
        [test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], 0, 1e-1) == -9648
    
    assert test_cell.total_charge( \
        [10, 10, 20], 0, 10) == 0 

    assert test_cell.total_charge( \
        [10, 10, 10], -10, 10) == 0 

def test_charge_conservation():
    """Tests that no charge disappear."""
    
    EPS = 1e-14

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 115.
    Cl_se0 = 148.
    Ca_si0 = 1.
    Ca_se0 = 1.
    Na_di0 = 15.
    Na_de0 = 145.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 115.
    Cl_de0 = 148.
    Ca_di0 = 1.
    Ca_de0 = 1.

    t_span = (0, 1000)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = sol.y
    t = sol.t
    
    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, 0)

    q_si = test_cell.total_charge([test_cell.Na_si[-1], test_cell.K_si[-1], test_cell.Cl_si[-1], test_cell.Cl_si[-1]], 0, test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.Na_se[-1], test_cell.K_se[-1], test_cell.Cl_se[-1], test_cell.Cl_se[-1]], 0, test_cell.V_se)        
    q_di = test_cell.total_charge([test_cell.Na_di[-1], test_cell.K_di[-1], test_cell.Cl_di[-1], test_cell.Cl_di[-1]], 0, test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.Na_de[-1], test_cell.K_de[-1], test_cell.Cl_de[-1], test_cell.Cl_de[-1]], 0, test_cell.V_de)

    total_q = abs(q_si + q_se + q_di + q_de)

    assert total_q < EPS

def test_charge_conservation_w_diffusion():
    """Tests that no charge disappear when ions diffuse."""
    
    EPS = 1e-14

    Na_si0 = 12.
    Na_se0 = 142.
    K_si0 = 99.
    K_se0 = 2.
    Cl_si0 = 111.
    Cl_se0 = 144.
    Ca_si0 = 1.
    Ca_se0 = 1.
    Na_di0 = 18.
    Na_de0 = 148.
    K_di0 = 101.
    K_de0 = 4.
    Cl_di0 = 119.
    Cl_de0 = 152.
    Ca_di0 = 2.
    Ca_de0 = 2.

    t_span = (0, 1000)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = sol.y
    t = sol.t
    
    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, 0)

    q_si = test_cell.total_charge([test_cell.Na_si[-1], test_cell.K_si[-1], test_cell.Cl_si[-1], test_cell.Cl_si[-1]], 0, test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.Na_se[-1], test_cell.K_se[-1], test_cell.Cl_se[-1], test_cell.Cl_se[-1]], 0, test_cell.V_se)        
    q_di = test_cell.total_charge([test_cell.Na_di[-1], test_cell.K_di[-1], test_cell.Cl_di[-1], test_cell.Cl_di[-1]], 0, test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.Na_de[-1], test_cell.K_de[-1], test_cell.Cl_de[-1], test_cell.Cl_de[-1]], 0, test_cell.V_de)

    total_q = abs(q_si + q_se + q_di + q_de)

    assert total_q < EPS

#def test_charge_symmetry():
#    """Tests that Q_i = -Q_e."""
#
#    EPS = 1e-14
#
#    Na_si0 = 15.
#    Na_se0 = 145.
#    K_si0 = 100.
#    K_se0 = 3.
#    Cl_si0 = 115.
#    Cl_se0 = 148.
#    Na_di0 = 15.
#    Na_de0 = 145.
#    K_di0 = 100.
#    K_de0 = 3.
#    Cl_di0 = 115.
#    Cl_de0 = 148.
#
#    t_span = (0, 100)
#    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]
#    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)
#
#    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
#    t = sol.t
#    
#    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
#        Cl_si, Cl_se, Cl_di, Cl_de)
#
#    q_si = test_cell.total_charge([test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], \
#        test_cell.V_si)
#    q_se = test_cell.total_charge([test_cell.Na_se, test_cell.K_se, test_cell.Cl_se], \
#        test_cell.V_se)
#    
#    q_di = test_cell.total_charge([test_cell.Na_di, test_cell.K_di, test_cell.Cl_di], \
#        test_cell.V_di)
#    q_de = test_cell.total_charge([test_cell.Na_de, test_cell.K_de, test_cell.Cl_de], \
#        test_cell.V_de)
#
#    N = len(q_si)
#    for i in range(N):
#        assert abs(q_si[i] + q_se[i]) < EPS
#        assert abs(q_di[i] + q_de[i]) < EPS
#
#def test_charge_symmetry_w_diffusion():
#    """Tests that Q_i = -Q_e when ions diffuse."""
#
#    EPS = 1e-14
#
#    Na_si0 = 12.
#    Na_se0 = 142.
#    K_si0 = 99.
#    K_se0 = 2.
#    Cl_si0 = 111.
#    Cl_se0 = 144.
#    Na_di0 = 18.
#    Na_de0 = 148.
#    K_di0 = 101.
#    K_de0 = 4.
#    Cl_di0 = 119.
#    Cl_de0 = 152.
#
#    t_span = (0, 100)
#    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]
#    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)
#
#    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
#    t = sol.t
#    
#    test_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
#        Cl_si, Cl_se, Cl_di, Cl_de)
#
#    q_si = test_cell.total_charge([test_cell.Na_si, test_cell.K_si, test_cell.Cl_si], \
#        test_cell.V_si)
#    q_se = test_cell.total_charge([test_cell.Na_se, test_cell.K_se, test_cell.Cl_se], \
#        test_cell.V_se)
#    
#    q_di = test_cell.total_charge([test_cell.Na_di, test_cell.K_di, test_cell.Cl_di], \
#        test_cell.V_di)
#    q_de = test_cell.total_charge([test_cell.Na_de, test_cell.K_de, test_cell.Cl_de], \
#        test_cell.V_de)
#
#    N = len(q_si)
#    for i in range(N):
#        assert abs(q_si[i] + q_se[i]) < EPS
#        assert abs(q_di[i] + q_de[i]) < EPS

