import numpy as np
from math import fsum
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class LeakyCell(): 

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de):
        self.T = T
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
        self.Cl_di = Cl_si 
        self.Cl_de = Cl_de

        self.C_sm = 3e-2
        self.C_dm = 3e-2
        self.A_s = 5e-10
        self.A_d = 5e-10 
        self.A_e = 1e-10
        self.A_i = 1e-10
        self.V_si = 1e-15
        self.V_se = 1e-15
        self.V_di = 1e-15
        self.V_de = 1e-15

        self.D_Na = 1.33e-9
        self.D_K = 1.96e-9 
        self.D_Cl = 2.03e-9

        self.dx = 10e-6

        self.lamda_i = 3.2
        self.lamda_e = 1.6

        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.

        self.F = 9.648e4    # [C  mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        self.g_Na_leak = 0.247    # [S m**-2]
        self.g_K_leak = 0.5
        self.g_Cl_leak = 1.0

        self.D_k = [self.D_Na, self.D_K, self.D_Cl]
        self.Z_k = [self.Z_Na, self.Z_K, self.Z_Cl]
        self.k_di = [Na_di, K_di, Cl_di]
        self.k_si = [Na_si, K_si, Cl_si]
        self.k_de = [Na_de, K_de, Cl_de]
        self.k_se = [Na_se, K_se, Cl_se]

    def nernst_potential(self, Z, k_i, k_e):
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        return E

    def I_diff(self, tortuosity, k_d, k_s):
        diff_sum = 0
        for k in range(0,len(self.D_k)):
            diff_sum += self.D_k[k] * self.Z_k[k] * (k_d[k] - k_s[k])
        I = self.F * diff_sum / (tortuosity**2 * self.dx)
        return I

    def sigma(self, tortuosity, k_d, k_s):
        sigma_sum = 0
        for k in range(0, len(self.D_k)):
            sigma_sum += self.D_k[k] * self.Z_k[k]**2 * (k_d[k] + k_s[k]) / 2
        s = self.F**2 * sigma_sum / (self.R*self.T*tortuosity**2)
        return s

    def nernst_planck(self, D, Z, k_d, k_s, tortuosity, phi_d, phi_s):
        j = -D * (k_d - k_s) / (tortuosity**2 * self.dx) - D*Z*self.F*(k_d + k_s)*(phi_d - phi_s)/(2*tortuosity**2*self.R*self.T*self.dx)
        return j

    def Q(self, k, V):
        q = 0
        for i in range(0, len(self.Z_k)):
            q += self.Z_k[i]*k[i]
        q = self.F*q*V
        return q

    def i_Na_s(self, phi_sm, E_Na_s):
        i_Na_s = self.g_Na_leak*(phi_sm - E_Na_s)
        return i_Na_s

    def i_K_s(self, phi_sm, E_K_s):
        i_K_s = self.g_K_leak*(phi_sm - E_K_s)
        return i_K_s

    def i_Cl_s(self, phi_sm, E_Cl_s):
        i_Cl_s = self.g_Cl_leak*(phi_sm - E_Cl_s)
        return i_Cl_s

    def i_Na_d(self, phi_dm, E_Na_d):
        i_Na_d = self.g_Na_leak*(phi_dm - E_Na_d)
        return i_Na_d

    def i_K_d(self, phi_dm, E_K_d):
        i_K_d = self.g_K_leak*(phi_dm - E_K_d)    
        return i_K_d

    def i_Cl_d(self, phi_dm, E_Cl_d):
        i_Cl_d = self.g_Cl_leak*(phi_dm - E_Cl_d)
        return i_Cl_d

    def membrane_potentials(self):

        I_i_diff = self.I_diff(self.lamda_i, self.k_di, self.k_si) 
        I_e_diff = self.I_diff(self.lamda_e, self.k_de, self.k_se) 
        sigma_i = self.sigma(self.lamda_i, self.k_di, self.k_si)
        sigma_e = self.sigma(self.lamda_e, self.k_de, self.k_se)

        Q_di = self.Q(self.k_di, self.V_di)
        Q_si = self.Q(self.k_si, self.V_si)

        phi_di = Q_di / (self.C_dm * self.A_d)
        phi_se = (self.dx * I_i_diff / sigma_i + self.A_e * self.V_si * self.dx * I_e_diff / (self.V_se * self.A_i * sigma_i) + phi_di - Q_si / (self.C_sm * self.A_s)) / (1 + self.A_e*self.V_si*sigma_e/(self.V_se*self.A_i*sigma_i))
        phi_si = Q_si / (self.C_sm * self.A_s) + phi_se
        phi_de = 0
        phi_sm = phi_si - phi_se
        phi_dm = phi_di - phi_de

        return phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm

    def reversal_potentials(self):
        
        E_Na_s = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_d = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_K_s = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_d = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_Cl_s = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_d = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)

        return E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d

    def dKdt(self):
       
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = self.membrane_potentials()
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = self.reversal_potentials()

        I_Na_s = self.i_Na_s(phi_sm, E_Na_s)
        I_K_s = self.i_K_s(phi_sm, E_K_s)
        I_Cl_s = self.i_Cl_s(phi_sm, E_Cl_s)

        I_Na_d = self.i_Na_d(phi_dm, E_Na_d)
        I_K_d = self.i_K_d(phi_dm, E_K_d)    
        I_Cl_d = self.i_Cl_d(phi_dm, E_Cl_d)

        j_Na_sm = I_Na_s / (self.F*self.Z_Na)
        j_K_sm = I_K_s / (self.F*self.Z_K) 
        j_Cl_sm = I_Cl_s / (self.F*self.Z_Cl)  

        j_Na_dm = I_Na_d / (self.F*self.Z_Na)
        j_K_dm = I_K_d / (self.F*self.Z_K) 
        j_Cl_dm = I_Cl_d / (self.F*self.Z_Cl)

        j_Na_i = self.nernst_planck(self.D_Na, self.Z_Na, self.Na_di, self.Na_si, self.lamda_i, phi_di, phi_si)
        j_K_i = self.nernst_planck(self.D_K, self.Z_K, self.K_di, self.K_si, self.lamda_i, phi_di, phi_si)
        j_Cl_i = self.nernst_planck(self.D_Cl, self.Z_Cl, self.Cl_di, self.Cl_si, self.lamda_i, phi_di, phi_si)

        j_Na_e = self.nernst_planck(self.D_Na, self.Z_Na, self.Na_de, self.Na_se, self.lamda_e, phi_de, phi_se)
        j_K_e = self.nernst_planck(self.D_K, self.Z_K, self.K_de, self.K_se, self.lamda_e, phi_de, phi_se)
        j_Cl_e = self.nernst_planck(self.D_Cl, self.Z_Cl, self.Cl_de, self.Cl_se, self.lamda_e, phi_de, phi_se)

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

    def hello(self):
        print "Hello World!"

if __name__ == "__main__":

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = k

        my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de = my_cell.dKdt()

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de
    
    start_time = time.time()
    t_span = (0, 10800)

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 115.
    Cl_se0 = 148.
    Na_di0 = 15.
    Na_de0 = 145.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 115.
    Cl_de0 = 148.

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]

    init_cell = LeakyCell(279.3, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0)
    
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

    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    t = sol.t

    my_cell = LeakyCell(279.3, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)
    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = my_cell.reversal_potentials()

    print 'elapsed time: ', time.time() - start_time

    plt.plot(t, phi_sm, '-', label='Vs')
    plt.plot(t, phi_dm, '-', label='Vd')
    plt.legend()
    plt.show()

    plt.plot(t, Na_si, label='Na_si')
    plt.plot(t, Na_se, label='Na_se')
    plt.plot(t, Na_di, label='Na_di')
    plt.plot(t, Na_de, label='Na_de')
    plt.plot(t, Na_si+Na_se+Na_di+Na_de, label='tot')
    plt.legend()
    plt.show()

    plt.plot(t, K_si, label='K_si')
    plt.plot(t, K_se, label='K_se')
    plt.plot(t, K_di, label='K_di')
    plt.plot(t, K_de, label='K_de')
    plt.plot(t, K_si+K_se+K_di+K_de, label='tot')
    plt.legend()
    plt.show()

    plt.plot(t, Cl_si, label='Cl_si')
    plt.plot(t, Cl_se, label='Cl_se')
    plt.plot(t, Cl_di, label='Cl_di')
    plt.plot(t, Cl_de, label='Cl_de')
    plt.plot(t, Cl_si+Cl_se+Cl_di+Cl_de, label='tot')
    plt.legend()
    plt.show()

