from pump import Pump
from somatic_injection_current import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import time

class PinskyRinzel(Pump):
    """ A two plus two compartment cell model with Pinsky-Rinzel mechanisms and pumps.

    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, Ca0_si, Ca0_di, n, h, s, c, q):

        Pump.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de)

        self.Ca0_si = Ca0_si
        self.Ca0_di = Ca0_di
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q

        # conductances [S * m**-2]
        self.g_Na = 300.
        self.g_DR = 150.
        self.g_Ca = 100.
        self.g_AHP = 8.
        self.g_C = 150.

    def alpha_m(self, phi_sm):
        phi_1 = phi_sm*1e3 + 46.9
        alpha = - 0.32 * phi_1 / (np.exp(-phi_1 / 4) - 1.)
        alpha = alpha*1e3
        return alpha

    def beta_m(self, phi_sm):
        phi_2 = phi_sm*1e3 + 19.9
        beta = 0.28 * phi_2 / (np.exp(phi_2 / 5.) - 1.)
        beta = beta*1e3
        return beta

    def alpha_h(self, phi_sm):
        alpha = 0.128 * np.exp((-43. - phi_sm*1e3) / 18.)
        alpha = alpha*1e3
        return alpha

    def beta_h(self, phi_sm):
        phi_5 = phi_sm*1e3 + 20.
        beta = 4. / (1 + np.exp(-phi_5 / 5.))
        beta = beta*1e3
        return beta

    def alpha_n(self, phi_sm):
        phi_3 = phi_sm*1e3 + 24.9
        alpha = - 0.016 * phi_3 / (np.exp(-phi_3 / 5.) - 1)
        alpha = alpha*1e3
        return alpha

    def beta_n(self, phi_sm):
        phi_4 = phi_sm*1e3 + 40.
        beta = 0.25 * np.exp(-phi_4 / 40.)
        beta = beta*1e3
        return beta

    def alpha_s(self, phi_dm):
        alpha = 1.6 / (1 + np.exp(-0.072 * (phi_dm*1e3 - 5.)))
        alpha = alpha*1e3
        return alpha

    def beta_s(self, phi_dm):
        phi_6 = phi_dm*1e3 + 8.9
        beta = 0.02 * phi_6 / (np.exp(phi_6 / 5.) - 1.)
        beta = beta*1e3
        return beta

    def alpha_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        phi_8 = phi_dm*1e3 + 50.0
        if phi_dm*1e3 <= -10:
            alpha = 0.0527 * np.exp(phi_8/11.- phi_7/27.)
        else:
            alpha = 2 * np.exp(-phi_7 / 27.)
        alpha = alpha*1e3
        return alpha

    def beta_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        if phi_dm*1e3 <= -10:
            beta = 2. * np.exp(-phi_7 / 27.) - self.alpha_c(phi_dm)/1e3
        else:
            beta = 0.
        beta = beta*1e3
        return beta

    def chi(self):
        return min(self.free_Ca_di/2.5e-4, 1.0)

    def alpha_q(self):
        return min(2e4*self.free_Ca_di, 10) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_sm):
        return self.alpha_m(phi_sm) / (self.alpha_m(phi_sm) + self.beta_m(phi_sm))

    def j_Na_s(self, phi_sm, E_Na_s):
        j = Pump.j_Na_s(self, phi_sm, E_Na_s) \
            + self.g_Na * self.m_inf(phi_sm)**2 * self.h * (phi_sm - E_Na_s) / (self.F*self.Z_Na)
        return j

    def j_K_s(self, phi_sm, E_K_s):
        j = Pump.j_K_s(self, phi_sm, E_K_s) \
            + self.g_DR * self.n * (phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = Pump.j_K_d(self, phi_dm, E_K_d) \
            + self.g_AHP * self.q * (phi_dm - E_K_d) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_dm - E_K_d) / (self.F*self.Z_K)
        return j

    def j_Ca_d(self, phi_dm, E_Ca_d):
        j = self.g_Ca * self.s**2 * (phi_dm - E_Ca_d) / (self.F*self.Z_Ca)
        return j

    def dkdt(self):

        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = Pump.membrane_potentials(self)
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = Pump.reversal_potentials(self)
        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = Pump.dkdt(self)

        V_fr_s = self.V_si/self.V_se
        V_fr_d = self.V_di/self.V_de 

        j_Ca_dm = self.j_Ca_d(phi_dm, E_Ca_d)

        dNadt_si = dNadt_si + 3*75*(self.Ca_si - self.Ca0_si)
        dNadt_se = dNadt_se - 3*75*V_fr_s*(self.Ca_si - self.Ca0_si)
        dNadt_di = dNadt_di + 3*75*(self.Ca_di - self.Ca0_di)
        dNadt_de = dNadt_de - 3*75*V_fr_d*(self.Ca_di - self.Ca0_di)

        dCadt_si = dCadt_si - 75*(self.Ca_si - self.Ca0_si)
        dCadt_se = dCadt_se + 75*V_fr_s*(self.Ca_si - self.Ca0_si)
        dCadt_di = dCadt_di - j_Ca_dm*(self.A_d / self.V_di) - 75*(self.Ca_di - self.Ca0_di)
        dCadt_de = dCadt_de + j_Ca_dm*(self.A_d / self.V_de) + 75*V_fr_d*(self.Ca_di - self.Ca0_di)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de

    def dmdt(self):
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = Pump.membrane_potentials(self)
        
        dndt = self.alpha_n(phi_sm)*(1-self.n) - self.beta_n(phi_sm)*self.n
        dhdt = self.alpha_h(phi_sm)*(1-self.h) - self.beta_h(phi_sm)*self.h 
        dsdt = self.alpha_s(phi_dm)*(1-self.s) - self.beta_s(phi_dm)*self.s
        dcdt = self.alpha_c(phi_dm)*(1-self.c) - self.beta_c(phi_dm)*self.c
        dqdt = self.alpha_q()*(1-self.q) - self.beta_q()*self.q
        
        return dndt, dhdt, dsdt, dcdt, dqdt

if __name__ == "__main__":

    T = 309.14

    Na_si0 = 18.
    K_si0 = 140.
    Cl_si0 = 6.
    Ca_si0 = 20*50e-6

    Na_se0 = 144.
    K_se0 = 4.
    Cl_se0 = 130.
    Ca_se0 = 1.1

    Na_di0 = 18.
    K_di0 = 140.
    Cl_di0 = 6.
    Ca_di0 = 20*50e-6

    Na_de0 = 144.
    K_de0 = 4.
    Cl_de0 = 130.
    Ca_de0 = 1.1

    k_res_si0 = Cl_si0 - (Na_si0 + K_si0 + 2*Ca_si0) - 0.035*3.5
    k_res_se0 = Cl_se0 - (Na_se0 + K_se0 + 2*Ca_se0) + 0.07*3.5
    k_res_di0 = Cl_di0 - (Na_di0 + K_di0 + 2*Ca_di0) - 0.035*3.5
    k_res_de0 = Cl_de0 - (Na_de0 + K_de0 + 2*Ca_de0) + 0.07*3.5

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01

    I_stim = 2000e-12 # [A]
    stim_dur = 0.035

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q = k

        my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, Ca_si0, Ca_di0, n, h, s, c, q)

        my_cell.A_i = my_cell.A_i*1e3
        my_cell.A_e = my_cell.A_e*1e3
        #my_cell.g_Ca = 400 
        #my_cell.g_DR = 100 
        #my_cell.g_AHP = 4
        #my_cell.g_Ca = 200

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt = my_cell.dmdt()

        if t > 1:
            dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, my_cell.Z_K, I_stim)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dresdt_si, dresdt_se, dresdt_di, dresdt_de, dndt, dhdt, dsdt, dcdt, dqdt
    
    start_time = time.time()
    t_span = (0, 1.5)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, n0, h0, s0, c0, q0]

    init_cell = PinskyRinzel(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si0, k_res_se0, k_res_di0, k_res_de0, Ca_si0, Ca_di0, n0, h0, s0, c0, q0)

    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = init_cell.reversal_potentials()

    print "----------------------------"
    print "Initial values"
    print "----------------------------"
    print 'phi_si: ', phi_si
    print 'phi_se: ', phi_se
    print 'phi_di: ', phi_di
    print 'phi_de: ', phi_de
    print 'phi_sm: ', phi_sm
    print 'phi_dm: ', phi_dm
    print 'E_Na_s: ', E_Na_s
    print 'E_Na_d: ', E_Na_d
    print 'E_K_s: ', E_K_s
    print 'E_K_d: ', E_K_d
    print 'E_Cl_s: ', E_Cl_s
    print 'E_Cl_d: ', E_Cl_d
    print 'E_Ca_s: ', E_Ca_s
    print 'E_Ca_d: ', E_Ca_d
    print "----------------------------"

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, n, h, s, c, q = sol.y
    t = sol.t

    my_cell = PinskyRinzel(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, Ca_si0, Ca_di0, n, h, s, c, q)
    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

    q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si[-1], my_cell.V_si)
    q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se[-1], my_cell.V_se)        
    q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di[-1], my_cell.V_di)
    q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de[-1], my_cell.V_de)
    print "total charge at the end (C): ", q_si + q_se + q_di + q_de

    print 'elapsed time: ', round(time.time() - start_time, 1), 'seconds'

    plt.plot(t, phi_sm, '-', label='Vs')
    plt.plot(t, phi_dm, '-', label='Vd')
    plt.title('Membrane potentials')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, E_Na_s, label='E_Na')
    plt.plot(t, E_K_s, label='E_K')
    plt.plot(t, E_Cl_s, label='E_Cl')
    plt.plot(t, E_Ca_s, label='E_Ca')
    plt.title('Reversal potentials soma')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, E_Na_d, label='E_Na')
    plt.plot(t, E_K_d, label='E_K')
    plt.plot(t, E_Cl_d, label='E_Cl')
    plt.plot(t, E_Ca_d, label='E_Ca')
    plt.title('Reversal potentials dendrite')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

#    plt.plot(t, Na_si, label='Na_si')
#    plt.plot(t, Na_se, label='Na_se')
#    plt.plot(t, Na_di, label='Na_di')
#    plt.plot(t, Na_de, label='Na_de')
#    plt.title('Sodium concentrations')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, K_si, label='K_si')
#    plt.plot(t, K_se, label='K_se')
#    plt.plot(t, K_di, label='K_di')
#    plt.plot(t, K_de, label='K_de')
#    plt.title('Potassium concentrations')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, Cl_si, label='Cl_si')
#    plt.plot(t, Cl_se, label='Cl_se')
#    plt.plot(t, Cl_di, label='Cl_di')
#    plt.plot(t, Cl_de, label='Cl_de')
#    plt.title('Chloride concentrations')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, Ca_si, label='Ca_si')
#    plt.plot(t, Ca_se, label='Ca_se')
#    plt.plot(t, Ca_di, label='Ca_di')
#    plt.plot(t, Ca_de, label='Ca_de')
#    plt.title('Calsium concentrations')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()
#
    plt.plot(t, q)
    plt.title('q')
    plt.xlabel('time [s]')
    plt.show()
##    plt.plot(t, my_cell.free_Ca_si, label='free_Ca_si')
    plt.plot(t, my_cell.free_Ca_di, label='free_Ca_di')
    plt.title('Free Calsium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()
