from pinsky_rinzel_pump.pinskyrinzel import *
from pinsky_rinzel_pump.somatic_injection_current import *

T = 309.14

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

k_res_si0 = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0 + res_i
k_res_se0 = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0 - res_e
k_res_di0 = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0 + res_i
k_res_de0 = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0 - res_e

n0 = 0.001
h0 = 0.999
s0 = 0.009
c0 = 0.007
q0 = 0.01

I_stim = 1e-12 # [A]

alpha = (12.5/12.5)*2.0

def dkdt(t,k):

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q = k

    my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q)


    dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
        dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
    dndt, dhdt, dsdt, dcdt, dqdt = my_cell.dmdt()

    dresdt_si, dresdt_se = somatic_injection_current(my_cell, dresdt_si, dresdt_se, 1.0, I_stim)

    return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
        dresdt_si, dresdt_se, dresdt_di, dresdt_de, dndt, dhdt, dsdt, dcdt, dqdt

start_time = time.time()
t_span = (0, 1)

k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, n0, h0, s0, c0, q0]

init_cell = PinskyRinzel(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, alpha, Ca_si0, Ca_di0, n0, h0, s0, c0, q0)


###
sigma_i = (init_cell.conductivity_k(init_cell.D_Na, init_cell.Z_Na, init_cell.lamda_i, init_cell.Na_si, init_cell.Na_di) \
    + init_cell.conductivity_k(init_cell.D_K, init_cell.Z_K, init_cell.lamda_i, init_cell.K_si, init_cell.K_di) \
    + init_cell.conductivity_k(init_cell.D_Cl, init_cell.Z_Cl, init_cell.lamda_i, init_cell.Cl_si, init_cell.Cl_di) \
    + init_cell.conductivity_k(init_cell.D_Ca, init_cell.Z_Ca, init_cell.lamda_i, init_cell.free_Ca_si, init_cell.free_Ca_di))/init_cell.dx
print("sigma:", sigma_i)
###

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
print("Q_se (C): ", q_se)
print("Q_di (C):", q_di)
print("Q_de (C): ", q_de)
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

sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q = sol.y
t = sol.t

my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q)

phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si[-1], my_cell.V_si)
q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se[-1], my_cell.V_se)        
q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di[-1], my_cell.V_di)
q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de[-1], my_cell.V_de)
print("Final values")
print("----------------------------")
print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
print("Q_si (C):", q_si)
print("Q_se (C): ", q_se)
print("Q_di (C):", q_di)
print("Q_de (C): ", q_de)

print("----------------------------")
print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

f1 = plt.figure(1)
plt.plot(t, phi_sm*1000, '-', label='V_s')
plt.plot(t, phi_dm*1000, '-', label='V_d')
plt.title('Membrane potentials')
plt.xlabel('time [s]')
plt.ylabel('[mV]')
plt.legend()

f2 = plt.figure(2)
plt.plot(t, phi_si, '-', label='V_si')
plt.plot(t, phi_se, '-', label='V_se')
plt.plot(t, phi_di, '-', label='V_di')
plt.plot(t, np.ones(len(t))*phi_de, '-', label='V_de')
plt.plot(t, phi_sm, '--', label='V_ms')
plt.plot(t, phi_dm, '--', label='V_md')
plt.title('Potentials')
plt.xlabel('time [s]')
plt.legend()

f3 = plt.figure(3)
plt.plot(t, E_Na_s, label='E_Na')
plt.plot(t, E_K_s, label='E_K')
plt.plot(t, E_Cl_s, label='E_Cl')
plt.plot(t, E_Ca_s, label='E_Ca')
plt.title('Reversal potentials soma')
plt.xlabel('time [s]')
plt.legend()

f4 = plt.figure(4)
plt.plot(t, E_Na_d, label='E_Na')
plt.plot(t, E_K_d, label='E_K')
plt.plot(t, E_Cl_d, label='E_Cl')
plt.plot(t, E_Ca_d, label='E_Ca')
plt.title('Reversal potentials dendrite')
plt.xlabel('time [s]')
plt.legend()

f5 = plt.figure(5)
plt.plot(t, Na_si, label='Na_si')
plt.plot(t, Na_se, label='Na_se')
plt.plot(t, Na_di, label='Na_di')
plt.plot(t, Na_de, label='Na_de')
plt.title('Sodium concentrations')
plt.xlabel('time [s]')
plt.legend()

f6 = plt.figure(6)
plt.plot(t, K_si, label='K_si')
plt.plot(t, K_se, label='K_se')
plt.plot(t, K_di, label='K_di')
plt.plot(t, K_de, label='K_de')
plt.title('Potassium concentrations')
plt.xlabel('time [s]')
plt.legend()

f7 = plt.figure(7)
plt.plot(t, Cl_si, label='Cl_si')
plt.plot(t, Cl_se, label='Cl_se')
plt.plot(t, Cl_di, label='Cl_di')
plt.plot(t, Cl_de, label='Cl_de')
plt.title('Chloride concentrations')
plt.xlabel('time [s]')
plt.legend()

f8 = plt.figure(8)
plt.plot(t, Ca_si, label='Ca_si')
plt.plot(t, Ca_se, label='Ca_se')
plt.plot(t, Ca_di, label='Ca_di')
plt.plot(t, Ca_de, label='Ca_de')
plt.title('Calsium concentrations')
plt.xlabel('time [s]')
plt.legend()

f9 = plt.figure(9)
plt.plot(t, q)
plt.title('q')
plt.xlabel('time [s]')

f9 = plt.figure(11)
plt.plot(t, s)
plt.title('s')
plt.xlabel('time [s]')

f10 = plt.figure(10)
plt.plot(t, my_cell.free_Ca_di*1e6, label='free_Ca_di')
plt.title('Free Calsium concentrations')
plt.xlabel('time [s]')
plt.ylabel('free [Ca]_i [nM]')
plt.legend()

plt.show()
