import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pinsky_rinzel_pump.pinskyrinzel import *
from pinsky_rinzel_pump.somatic_injection_current import *

def pinskyrinzel(I_stim, g_Na, g_DR):

    T = 309.14

    Na_si0 = 18.
    Na_se0 = 144.
    K_si0 = 140.
    K_se0 = 4.
    Cl_si0 = 6.
    Cl_se0 = 130.
    Ca_si0 = 20*50e-6
    Ca_se0 = 1.1
    Na_di0 = 18.
    Na_de0 = 144.
    K_di0 = 140.
    K_de0 = 4.
    Cl_di0 = 6.
    Cl_de0 = 130.
    Ca_di0 = 20*50e-6
    Ca_de0 = 1.1
    k_res_si = Cl_si0 - (Na_si0 + K_si0 + 2*Ca_si0) - 0.035*3.5
    k_res_se = Cl_se0 - (Na_se0 + K_se0 + 2*Ca_se0) + 0.07*3.5
    k_res_di = Cl_di0 - (Na_di0 + K_di0 + 2*Ca_di0) - 0.035*3.5
    k_res_de = Cl_de0 - (Na_de0 + K_de0 + 2*Ca_de0) + 0.07*3.5

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01

    def dkdt(t, k, I_stim, g_Na, g_DR):
        
        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
            Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q = k

        my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
            Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, Ca_si0, Ca_di0, n, h, s, c, q)
        
        my_cell.A_i = 3e-9
        my_cell.A_e = 3e-9/2.0
        my_cell.g_Na = g_Na
        my_cell.g_DR = g_DR

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt = my_cell.dmdt()

        if t > 1:
            dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, my_cell.Z_K, I_stim)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt

    t_span = (0, 2)
    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, \
        n0, h0, s0, c0, q0]
    
    sol = solve_ivp(lambda t, k: dkdt(t, k, I_stim, g_Na, g_DR), t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, \
        n, h, s, c, q = sol.y
    time = sol.t

    my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, Ca_si0, Ca_di0, n, h, s, c, q)
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

    info = {"stimulus_start": 1.0, "stimulus_end": 2.0}
    return time, phi_sm, info

if __name__ == "__main__":
    time, phi, info = pinskyrinzel(1575e-12, 315)
    import matplotlib.pyplot as plt
    plt.plot(time, phi)
    plt.show()

