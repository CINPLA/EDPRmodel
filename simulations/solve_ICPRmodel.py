from ICPRmodel.ICPRmodel import *
from ICPRmodel.somatic_injection_current import *
from scipy.integrate import solve_ivp

def solve_ICPRmodel(t_dur, alpha, I_stim, stim_start, stim_end):

    T = 309.14

    Na_si0 = 18.
    Na_se0 = 140.
    K_si0 = 99.
    K_se0 = 4.3
    Cl_si0 = 7.
    Cl_se0 = 134.
    Ca_si0 = 0.01
    Ca_se0 = 1.1

    Na_di0 = 18.
    Na_de0 = 140.
    K_di0 = 99.
    K_de0 = 4.3
    Cl_di0 = 7.
    Cl_de0 = 134.
    Ca_di0 = 0.01
    Ca_de0 = 1.1

    res_i = -68e-3*3e-2*616e-12/(1437e-18*9.648e4)
    res_e = -68e-3*3e-2*616e-12/(718.5e-18*9.648e4)

    k_res_si0 = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0 + res_i
    k_res_se0 = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0 - res_e
    k_res_di0 = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0 + res_i
    k_res_de0 = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0 - res_e

    n0 = 0.0003
    h0 = 0.999
    s0 = 0.007
    c0 = 0.006
    q0 = 0.011
    z0 = 1.0

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q, z = k

        my_cell = ICPRmodel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()

        if t > stim_start and t < stim_end:
            dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, 1.0, I_stim)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dresdt_si, dresdt_se, dresdt_di, dresdt_de, dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    # calibrate 
    t_span = (0, t_dur)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, \
        k_res_si0, k_res_se0, k_res_di0, k_res_de0, n0, h0, s0, c0, q0, z0]

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)
    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, \
        k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q, z = sol.y

    # solve
    init_cell = ICPRmodel(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, alpha, Ca_si0, Ca_di0, n0, h0, s0, c0, q0, z0)
    
    ###################### 
    # print initial values    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()

    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = init_cell.reversal_potentials()

    q_si = init_cell.total_charge([init_cell.Na_si, init_cell.K_si, init_cell.Cl_si, init_cell.Ca_si], init_cell.k_res_si, init_cell.V_si)
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se], init_cell.k_res_se, init_cell.V_se)        
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di, init_cell.Ca_di], init_cell.k_res_di, init_cell.V_di)
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de], init_cell.k_res_de, init_cell.V_de)

    print("----------------------------")
    print("Initial values")
    print("----------------------------")
    print("initial total charge(C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C):", q_si)
    print("Q_se (C):", q_se)
    print("Q_di (C):", q_di)
    print("Q_de (C):", q_de)
    print("----------------------------")
    print("potentials [mV]")
    print('phi_si: ', round(phi_si*1000))
    print('phi_se: ', round(phi_se*1000))
    print('phi_di: ', round(phi_di*1000))
    print('phi_de: ', round(phi_de*1000))
    print('phi_sm: ', round(phi_sm*1000))
    print('phi_dm: ', round(phi_dm*1000))
    print('E_Na_s: ', round(E_Na_s*1000))
    print('E_Na_d: ', round(E_Na_d*1000))
    print('E_K_s: ', round(E_K_s*1000))
    print('E_K_d: ', round(E_K_d*1000))
    print('E_Cl_s: ', round(E_Cl_s*1000))
    print('E_Cl_d: ', round(E_Cl_d*1000))
    print('E_Ca_s: ', round(E_Ca_s*1000))
    print('E_Ca_d: ', round(E_Ca_d*1000))
    print("----------------------------")

    return sol
