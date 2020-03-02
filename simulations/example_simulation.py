import numpy as np
import time
import matplotlib.pyplot as plt
from EDPRmodel.EDPRmodel import *
from solve_EDPRmodel import solve_EDPRmodel

start_time = time.time()

t_dur = 30      # [s]
alpha = 2.0
I_stim = 28e-12 # [A]
stim_start = 10 # [s]
stim_end = 20   # [s]

sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end)

Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, \
    X_si, X_se, X_di, X_de, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()

E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1], my_cell.X_si[-1]], my_cell.V_si)
q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1], my_cell.X_se[-1]], my_cell.V_se)        
q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1], my_cell.X_di[-1]], my_cell.V_di)
q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1], my_cell.X_de[-1]], my_cell.V_de)
print("Final values")
print("----------------------------")
print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
print("Q_si (C):", q_si)
print("Q_se (C):", q_se)
print("Q_di (C):", q_di)
print("Q_de (C):", q_de)
print("----------------------------")
print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

f1 = plt.figure(1)
plt.plot(t, phi_sm*1000, '-', label='V_s')
plt.plot(t, phi_dm*1000, '-', label='V_d')
plt.title('Membrane potentials')
plt.xlabel('time [s]')
plt.ylabel('[mV]')
plt.legend(loc='upper right')

plt.show()
