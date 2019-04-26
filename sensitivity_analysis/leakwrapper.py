import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pinsky_rinzel_pump.leakycell import *

def leak_wrap(g_Na, g_K, g_Cl):

    T = 309.14
    alpha = 2.0

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 5.
    Cl_se0 = 134.
    Ca_si0 = 0
    Ca_se0 = 0
    Na_di0 = 15.
    Na_de0 = 145.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 5.
    Cl_de0 = 134.
    Ca_di0 = 0
    Ca_de0 = 0
    k_res_si = Cl_si0 - (Na_si0 + K_si0 + 2*Ca_si0)
    k_res_se = Cl_se0 - (Na_se0 + K_se0 + 2*Ca_se0)
    k_res_di = Cl_di0 - (Na_di0 + K_di0 + 2*Ca_di0)
    k_res_de = Cl_de0 - (Na_de0 + K_de0 + 2*Ca_de0)

    def dkdt(t, k, g_Na, g_K, g_Cl):
        
        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = k

        my_cell = LeakyCell(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
            Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
        
        my_cell.g_Na_leak = g_Na
        my_cell.g_K_leak = g_K
        my_cell.g_Cl_leak = g_Cl

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de

    t_span = (0, 5)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0]
    
    sol = solve_ivp(lambda t, k: dkdt(t, k, g_Na, g_K, g_Cl), t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = sol.y
    time = sol.t

    my_cell = LeakyCell(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

    return time, phi_sm

