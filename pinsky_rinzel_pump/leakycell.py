import numpy as np
from math import fsum
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class LeakyCell(): 
    """A four compartment cell model (soma + dendrite, both internal and external space) 
    with Na, K, and Cl leak currents.

    Methods:
        constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)
        j_Na_s(phi_sm, E_Na_s): compute the Na flux across the soma membrane
        j_K_s(phi_sm, E_K_s): compute the K flux across the soma membrane
        j_Cl_s(phi_sm, E_Cl_s): compute the Cl flux across the soma membrane
        j_Na_d(phi_dm, E_Na_d): compute the Na flux across the dendrite membrane
        j_K_d(phi_dm, E_K_d): compute the K flux across the dendrite membrane
        j_Cl_d(phi_dm, E_Cl_d): compute the Cl flux across the dendrite membrane
        j_k_diff(D_k, tortuosity, k_s, k_d): compute the axial diffusion flux of ion k
        j_k_drift(D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d): compute the axial drift flux of ion k
        conductivity_k(D_k, Z_k, tortuosity, k_s, k_d): compute axial conductivity of ion k
        total_charge(k, V): calculate the total charge within volume V
        nernst_potential(Z, k_i, k_e): calculate the reversal potential of ion k
        reversal_potentials(): calculate the reversal potentials of all ion species
        membrane_potentials(): calculate the membrane potentials
        dkdt(): calculate dk/dt for all ion species k
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de):
        
        # temperature [K]
        self.T = T

        # ion concentraions [mol * m**-2 * s**-1]
        self.Na_si = Na_si
        self.Na_se = Na_se
        self.Na_di = Na_di
        self.Na_de = Na_de
        self.K_si = K_si
        self.K_se = K_se
        self.K_di = K_di
        self.K_de = K_de
        self.Cl_si = Cl_si
        self.Cl_se = Cl_se 
        self.Cl_di = Cl_di 
        self.Cl_de = Cl_de

        # membrane capacitance [F * m**-2]
        self.C_sm = 3e-2 # Pinsky and Rinzel, 1994
        self.C_dm = 3e-2 # Pinsky and Rinzel, 1994
       
        # volumes and areas [m]
        self.A_s = 9e-6
        self.A_d = 9e-6 
        self.A_i = 3e-6
        self.A_e = 3e-6
        self.V_si = 2e-6
        self.V_di = 2e-6
        self.V_se = 1e-6
        self.V_de = 1e-6
        self.dx = 667e-6

        # diffusion constants
        self.D_Na = 1.33e-9 # Halnes et al. 2013
        self.D_K = 1.96e-9  # Halnes et al. 2013 
        self.D_Cl = 2.03e-9 # Halnes et al. 2013

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.

        # constants
        self.F = 9.648e4    # [C * mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        # conductances [S * m**-2]
        self.g_Na_leak = 0.247 # Wei et al. 2014
        self.g_K_leak = 0.5    # Wei et al. 2014
        self.g_Cl_leak = 1.0   # Wei et al. 2014

    def j_Na_s(self, phi_sm, E_Na_s):
        i_Na_s = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na)
        return i_Na_s

    def j_K_s(self, phi_sm, E_K_s):
        i_K_s = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K)
        return i_K_s

    def j_Cl_s(self, phi_sm, E_Cl_s):
        i_Cl_s = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl)
        return i_Cl_s

    def j_Na_d(self, phi_dm, E_Na_d):
        i_Na_d = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na)
        return i_Na_d

    def j_K_d(self, phi_dm, E_K_d):
        i_K_d = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K)    
        return i_K_d

    def j_Cl_d(self, phi_dm, E_Cl_d):
        i_Cl_d = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl)
        return i_Cl_d

    def j_k_diff(self, D_k, tortuosity, k_s, k_d):
        j = - D_k * (k_d - k_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (k_d + k_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, k_s, k_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (k_d + k_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k, V):
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl]
        q = 0
        for i in range(0, 3):
            q += Z_k[i]*k[i]
        q = self.F*q*V
        return q

    def nernst_potential(self, Z, k_i, k_e):
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        return E

    def reversal_potentials(self):
        E_Na_s = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_d = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_K_s = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_d = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_Cl_s = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_d = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)
        return E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d

    def membrane_potentials(self):
        I_i_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de)

        q_di = self.total_charge([self.Na_di, self.K_di, self.Cl_di], self.V_di)
        q_si = self.total_charge([self.Na_si, self.K_si, self.Cl_si], self.V_si)

        phi_di = q_di / (self.C_dm * self.A_d)
        phi_se = (phi_di - self.dx * I_i_diff / sigma_i - self.A_e * self.V_si * self.dx * I_e_diff / (self.V_se * self.A_i * sigma_i) - q_si / (self.C_sm * self.A_s)) / (1 + self.A_e*self.V_si*sigma_e/(self.V_se*self.A_i*sigma_i))
        phi_si = q_si / (self.C_sm * self.A_s) + phi_se
        phi_de = 0
        phi_sm = phi_si - phi_se
        phi_dm = phi_di - phi_de

        return phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm

    def dkdt(self):
       
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = self.membrane_potentials()
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = self.reversal_potentials()

        j_Na_sm = self.j_Na_s(phi_sm, E_Na_s)
        j_K_sm = self.j_K_s(phi_sm, E_K_s)
        j_Cl_sm = self.j_Cl_s(phi_sm, E_Cl_s)

        j_Na_dm = self.j_Na_s(phi_dm, E_Na_d)
        j_K_dm = self.j_K_d(phi_dm, E_K_d)    
        j_Cl_dm = self.j_Cl_d(phi_dm, E_Cl_d)

        j_Na_i = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di, phi_si, phi_di) 
        j_K_i = self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di, phi_si, phi_di)
        j_Cl_i = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di, phi_si, phi_di)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de, phi_se, phi_de)

        dNadt_si = -j_Na_sm*(self.A_s / self.V_si) - j_Na_i*(self.A_i / self.V_si)
        dNadt_di = -j_Na_dm*(self.A_d / self.V_di) + j_Na_i*(self.A_i / self.V_di)
        dNadt_se = j_Na_sm*(self.A_s / self.V_se) - j_Na_e*(self.A_e / self.V_se)
        dNadt_de = j_Na_dm*(self.A_d / self.V_de) + j_Na_e*(self.A_e / self.V_de)

        dKdt_si = -j_K_sm*(self.A_s / self.V_si) - j_K_i*(self.A_i / self.V_si)
        dKdt_di = -j_K_dm*(self.A_d / self.V_di) + j_K_i*(self.A_i / self.V_di)
        dKdt_se = j_K_sm*(self.A_s / self.V_se) - j_K_e*(self.A_e / self.V_se)
        dKdt_de = j_K_dm*(self.A_d / self.V_de) + j_K_e*(self.A_e / self.V_de)

        dCldt_si = -j_Cl_sm*(self.A_s / self.V_si) - j_Cl_i*(self.A_i / self.V_si)
        dCldt_di = -j_Cl_dm*(self.A_d / self.V_di) + j_Cl_i*(self.A_i / self.V_di)
        dCldt_se = j_Cl_sm*(self.A_s / self.V_se) - j_Cl_e*(self.A_e / self.V_se)
        dCldt_de = j_Cl_dm*(self.A_d / self.V_de) + j_Cl_e*(self.A_e / self.V_de)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de

if __name__ == "__main__":

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = k

        my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de = my_cell.dkdt()

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de
    
    start_time = time.time()
    t_span = (0, 10)

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

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]

    init_cell = LeakyCell(279.3, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0)
   
    print init_cell.Na_si
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di], init_cell.V_di)
    print 'Q_di: ', q_di

    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = init_cell.reversal_potentials()

    print 'phi_si: ', phi_si
    print 'phi_se: ', phi_se
    print 'phi_di: ', phi_di
    print 'phi_de: ', phi_de
    print 'phi_sm: ', phi_sm
    print 'phi_dm: ', phi_dm
    print 'E_Na_s: ', E_Na_s
    print 'E_K_s: ', E_K_s
    print 'E_K_s: ', E_K_d
    print 'E_Cl_s: ', E_Cl_s
    print 'E_Cl_d:', E_Cl_d

    #sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    #Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    #t = sol.t

#    my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)
#    
#    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
#    
#    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = my_cell.reversal_potentials()
#
#    print 'elapsed time: ', time.time() - start_time
#
#    plt.plot(t, phi_sm, '-', label='Vs')
#    plt.plot(t, phi_dm, '-', label='Vd')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, Na_si, label='Na_si')
#    plt.plot(t, Na_se, label='Na_se')
#    plt.plot(t, Na_di, label='Na_di')
#    plt.plot(t, Na_de, label='Na_de')
#    plt.plot(t, Na_si+Na_se+Na_di+Na_de, label='tot')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, K_si, label='K_si')
#    plt.plot(t, K_se, label='K_se')
#    plt.plot(t, K_di, label='K_di')
#    plt.plot(t, K_de, label='K_de')
#    plt.plot(t, K_si+K_se+K_di+K_de, label='tot')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, Cl_si, label='Cl_si')
#    plt.plot(t, Cl_se, label='Cl_se')
#    plt.plot(t, Cl_di, label='Cl_di')
#    plt.plot(t, Cl_de, label='Cl_de')
#    plt.plot(t, Cl_si+Cl_se+Cl_di+Cl_de, label='tot')
#    plt.legend()
#    plt.show()
#
