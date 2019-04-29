import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pinsky_rinzel_pump.pinskyrinzel import *
from pinsky_rinzel_pump.somatic_injection_current import *

def pinskyrinzel(g_Na, g_DR):

    T = 309.14
    alpha = 2.0

    Na_si0 = 18.
    Na_se0 = 160.
    K_si0 = 100.
    K_se0 = 6.
    Cl_si0 = 8.
    Cl_se0 = 100.
    Ca_si0 = 20*100e-6
    Ca_se0 = 1.1

    Na_di0 = 18.
    Na_de0 = 160.
    K_di0 = 100.
    K_de0 = 6.
    Cl_di0 = 8.
    Cl_de0 = 100.
    Ca_di0 = 20*100e-6
    Ca_de0 = 1.1

    res_i = -65e-3*3e-2*200e-12/(4000e-18*9.648e4)
    res_e = -65e-3*3e-2*200e-12/(2000e-18*9.648e4)
    k_res_si = Cl_si0 - (Na_si0 + K_si0 + 2*Ca_si0) + res_i
    k_res_se = Cl_se0 - (Na_se0 + K_se0 + 2*Ca_se0) - res_e
    k_res_di = Cl_di0 - (Na_di0 + K_di0 + 2*Ca_di0) + res_i
    k_res_de = Cl_de0 - (Na_de0 + K_de0 + 2*Ca_de0) - res_e

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01

    I_stim = 1e-12

    def dkdt(t, k, g_Na, g_DR):
        
        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
            Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q = k

        my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
            Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q)
        
        my_cell.g_Na = g_Na
        my_cell.g_DR = g_DR

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt = my_cell.dmdt()

        dresdt_si, dresdt_se = somatic_injection_current(my_cell, dresdt_si, dresdt_se, 1.0, I_stim)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt

    t_span = (0, 1)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, \
        n0, h0, s0, c0, q0]
    
    sol = solve_ivp(lambda t, k: dkdt(t, k, g_Na, g_DR), t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, \
        n, h, s, c, q = sol.y
    time = sol.t

    my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q)
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

    info = {"stimulus_start": 0.0, "stimulus_end": 1.0}
    return time, phi_sm, info

if __name__ == "__main__":
    time, phi, info = pinskyrinzel(315, 150)
    import matplotlib.pyplot as plt
    plt.plot(time, phi)
    plt.show()

