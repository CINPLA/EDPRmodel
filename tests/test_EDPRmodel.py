import pytest
from scipy.integrate import solve_ivp
from EDPRmodel.EDPRmodel import *
from EDPRmodel.somatic_injection_current import *

def test_EDPRmodel():
    """Tests that no charge disappear.
       Tests charge symmetry."""
   
    alpha = 1.0
    
    EPS = 1e-14

    Na_si0 = 18.
    Na_se0 = 144.
    K_si0 = 140.
    K_se0 = 4.
    Cl_si0 = 6.
    Cl_se0 = 130.
    Ca_si0 = 0.01
    Ca_se0 = 1.1

    Na_di0 = 18.
    Na_de0 = 144.
    K_di0 = 140.
    K_de0 = 4.
    Cl_di0 = 6.
    Cl_de0 = 130.
    Ca_di0 = 0.01
    Ca_de0 = 1.1

    X_si = Na_si0 + K_si0 - Cl_si0 + 2*Ca_si0
    X_se = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0
    X_di = Na_di0 + K_di0 - Cl_di0 + 2*Ca_di0
    X_de = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 10e-12 # [A]

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z = k

        my_cell = EDPRmodel(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
            Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()

        if t > 3 and t < 3.5:
            dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, my_cell.Z_K, I_stim)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    t_span = (0, 10)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, n0, h0, s0, c0, q0, z0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z = sol.y
    
    test_cell = EDPRmodel(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, \
        Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

    q_si = test_cell.total_charge([test_cell.Na_si[-1], test_cell.K_si[-1], test_cell.Cl_si[-1], test_cell.Ca_si[-1], test_cell.X_si], test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.Na_se[-1], test_cell.K_se[-1], test_cell.Cl_se[-1], test_cell.Ca_se[-1], test_cell.X_se], test_cell.V_se)        
    q_di = test_cell.total_charge([test_cell.Na_di[-1], test_cell.K_di[-1], test_cell.Cl_di[-1], test_cell.Ca_di[-1], test_cell.X_di], test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.Na_de[-1], test_cell.K_de[-1], test_cell.Cl_de[-1], test_cell.Ca_de[-1], test_cell.X_de], test_cell.V_de)

    total_q = abs(q_si + q_se + q_di + q_de)

    assert total_q < EPS
    assert abs(q_si + q_se) < EPS
    assert abs(q_di + q_de) < EPS
