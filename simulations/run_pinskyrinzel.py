from pinsky_rinzel_pump.pinskyrinzel import *
from pinsky_rinzel_pump.somatic_injection_current import *

T = 309.14

Na_si0 = 18
Na_se0 = 139
K_si0 = 99
K_se0 = 5
Cl_si0 = 7
Cl_se0 = 131
Ca_si0 = 0.01
Ca_se0 = 1.1

Na_di0 = 18
Na_de0 = 139
K_di0 = 99
K_de0 = 5
Cl_di0 = 7
Cl_de0 = 131
Ca_di0 = 0.01
Ca_de0 = 1.1

res_i = -66e-3*3e-2*616e-12/(1437e-18*9.648e4)
res_e = -66e-3*3e-2*616e-12/(718.5e-18*9.648e4)

k_res_si0 = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0 + res_i
k_res_se0 = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0 - res_e
k_res_di0 = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0 + res_i
k_res_de0 = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0 - res_e

n0 = 0.0004
h0 = 0.999
s0 = 0.008
c0 = 0.006
q0 = 0.011
z0 = 1.0

#I_stim = 35e-12 # [A]
#alpha = (2.6/12.5)*2.0

I_stim = 20e-12 # [A]
#I_stim = 40e-12 # [A]
alpha = (12.5/12.5)*2.0

def dkdt(t,k):

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q, z = k

    my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

    #my_cell.g_Ca = 119
    my_cell.g_Ca = 118

    dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
        dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
    dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()

    #if t < 7:
    #dresdt_si, dresdt_se = somatic_injection_current(my_cell, dresdt_si, dresdt_se, 1.0, I_stim)
    dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, 1.0, I_stim)

    return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
        dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
        dresdt_si, dresdt_se, dresdt_di, dresdt_de, dndt, dhdt, dsdt, dcdt, dqdt, dzdt

start_time = time.time()
t_span = (0, 30)

k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, n0, h0, s0, c0, q0, z0]

init_cell = PinskyRinzel(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, alpha, Ca_si0, Ca_di0, n0, h0, s0, c0, q0, z0)

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

sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

#np.savez('figure1_MJ', t=t, phi_sm=phi_sm, phi_dm=phi_dm, q=q, free_Ca_di=my_cell.free_Ca_di)

q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si[-1], my_cell.V_si)
q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se[-1], my_cell.V_se)        
q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di[-1], my_cell.V_di)
q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de[-1], my_cell.V_de)
print("Final values")
print("----------------------------")
print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
print("Q_si (C):", q_si)
print("Q_se (C):", q_se)
print("Q_di (C):", q_di)
print("Q_de (C):", q_de)

print("----------------------------")
print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')
print("----------------------------")

print('Na_se:', round(Na_se[-1],2))
print('Na_si:', round(Na_si[-1],2))
print('K_se:', round(K_se[-1],2))
print('K_si:', round(K_si[-1],2))
print('Cl_se:', round(Cl_se[-1],2))
print('Cl_si:', round(Cl_si[-1],2))
print('Ca_se:', round(Ca_se[-1],2))
print('Ca_si:', round(Ca_si[-1],2))
print('n:', round(n[-1],5))
print('h:', round(h[-1],5))
print('s:', round(s[-1],5))
print('c:', round(c[-1],5))
print('q:', round(q[-1],5))
print('z:', round(z[-1],5))

f1 = plt.figure(1)
plt.plot(t, phi_sm*1000, '-', label='V_s')
plt.plot(t, phi_dm*1000, '-', label='V_d')
plt.title('Membrane potentials')
plt.xlabel('time [s]')
plt.ylabel('[mV]')
plt.legend(loc='upper right')
#plt.ylim(-70, 10)


#f2 = plt.figure(2)
##plt.plot(t, phi_si, '-', label='V_si')
#plt.plot(t, phi_se, '-', label='V_se')
##plt.plot(t, phi_di, '-', label='V_di')
##plt.plot(t, np.ones(len(t))*phi_de, '-', label='V_de')
##plt.plot(t, phi_sm, '--', label='V_ms')
##plt.plot(t, phi_dm, '--', label='V_md')
#plt.title('Potentials')
#plt.xlabel('time [s]')
#plt.legend(loc='upper right')

f3 = plt.figure(3)
plt.plot(t, E_Na_s, label='E_Na')
plt.plot(t, E_K_s, label='E_K')
plt.plot(t, E_Cl_s, label='E_Cl')
plt.plot(t, E_Ca_s, label='E_Ca')
plt.title('Reversal potentials soma')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

f4 = plt.figure(4)
plt.plot(t, E_Na_d, label='E_Na')
plt.plot(t, E_K_d, label='E_K')
plt.plot(t, E_Cl_d, label='E_Cl')
plt.plot(t, E_Ca_d, label='E_Ca')
plt.title('Reversal potentials dendrite')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

f5 = plt.figure(5)
plt.plot(t, Na_si, label='Na_si')
plt.plot(t, Na_se, label='Na_se')
plt.plot(t, Na_di, label='Na_di')
plt.plot(t, Na_de, label='Na_de')
plt.title('Sodium concentrations')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

f6 = plt.figure(6)
plt.plot(t, K_si, label='K_si')
plt.plot(t, K_se, label='K_se')
plt.plot(t, K_di, label='K_di')
plt.plot(t, K_de, label='K_de')
plt.title('Potassium concentrations')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

f7 = plt.figure(7)
plt.plot(t, Cl_si, label='Cl_si')
plt.plot(t, Cl_se, label='Cl_se')
plt.plot(t, Cl_di, label='Cl_di')
plt.plot(t, Cl_de, label='Cl_de')
plt.title('Chloride concentrations')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

f8 = plt.figure(8)
plt.plot(t, Ca_si, label='Ca_si')
plt.plot(t, Ca_se, label='Ca_se')
plt.plot(t, Ca_di, label='Ca_di')
plt.plot(t, Ca_de, label='Ca_de')
plt.title('Calsium concentrations')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

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
plt.legend(loc='upper right')

plt.show()
