from leakycell import LeakyCell
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class Pump(LeakyCell):
    """A two plus two compartment cell model with Na, K and Cl leak currents, and pumps.

    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de, I_stim):
        LeakyCell.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de, I_stim)
        self.rho = 1.87e-6
        self.U_kcc2 = 7.00e-7
        self.U_nkcc1 = 2.33e-7
#        self.rho = 1.78e-7
#        self.U_kcc2 = 6.67e-8
#        self.U_nkcc1 = 2.22e-8

    def j_pump(self, Na_i, K_e):
        j = (self.rho / (1.0 + np.exp((25. - Na_i)/3.))) * (1.0 / (1.0 + np.exp(3.5 - K_e)))
        return j

    def j_kcc2(self, K_i, K_e, Cl_i, Cl_e):
        j = self.U_kcc2 * np.log(K_i*Cl_i/(K_e*Cl_e))
        return j
    
    def j_nkcc1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp(16 - K_e))) * (np.log(K_i*Cl_i/(K_e*Cl_e)) + np.log(Na_i*Cl_i/(Na_e*Cl_e)))
        return j

    def j_Na_s(self, phi_sm, E_Na_s):
        j = LeakyCell.j_Na_s(self, phi_sm, E_Na_s) \
            + 3*self.j_pump(self.Na_si, self.K_se) \
            - self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
       
    def j_K_s(self, phi_sm, E_K_s):
        j = LeakyCell.j_K_s(self, phi_sm, E_K_s) \
            - 2*self.j_pump(self.Na_si, self.K_se) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            - self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = LeakyCell.j_Cl_s(self, phi_sm, E_Cl_s) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            - 2*self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
            
    def j_Na_d(self, phi_dm, E_Na_d):
        j = LeakyCell.j_Na_d(self, phi_dm, E_Na_d) \
            + 3*self.j_pump(self.Na_di, self.K_de) \
            - self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = LeakyCell.j_K_d(self, phi_dm, E_K_d) \
            - 2*self.j_pump(self.Na_di, self.K_de) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            - self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = LeakyCell.j_Cl_d(self, phi_dm, E_Cl_d) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            - 2*self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

if __name__ == "__main__":

    T = 309.14

    Na_si0 = 18.
    K_si0 = 140.
    Cl_si0 = 6.

    Na_se0 = 144.
    K_se0 = 4.
    Cl_se0 = 130.

    Na_di0 = 18.
    K_di0 = 140.
    Cl_di0 = 6.

    Na_de0 = 144.
    K_de0 = 4.
    Cl_de0 = 130.

    k_rest_si = Cl_si0 - (Na_si0 + K_si0)#-0.035
    k_rest_se = Cl_se0 - (Na_se0 + K_se0)#+0.07
    k_rest_di = Cl_di0 - (Na_di0 + K_di0)#-0.035
    k_rest_de = Cl_de0 - (Na_de0 + K_de0)#+0.07

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = k

        my_cell = Pump(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de, 0)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de = my_cell.dkdt()

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de
    
    start_time = time.time()
    t_span = (0, 140)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0]

    init_cell = Pump(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, k_rest_si, k_rest_se, k_rest_di, k_rest_de, 0)

    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = init_cell.reversal_potentials()

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
    print 'E_Cl_d:', E_Cl_d
    print "----------------------------"

    sol = solve_ivp(dkdt, t_span, k0, max_step=0.001)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de = sol.y
    t = sol.t

    my_cell = Pump(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de, 0)
    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d = my_cell.reversal_potentials()

    q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1]], my_cell.k_rest_si, my_cell.V_si)
    q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1]], my_cell.k_rest_se, my_cell.V_se)        
    q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1]], my_cell.k_rest_di, my_cell.V_di)
    q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1]], my_cell.k_rest_de, my_cell.V_de)
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
    plt.title('Reversal potentials soma')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, E_Na_d, label='E_Na')
    plt.plot(t, E_K_d, label='E_K')
    plt.plot(t, E_Cl_d, label='E_Cl')
    plt.title('Reversal potentials dendrite')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, Na_si, label='Na_si')
    plt.plot(t, Na_se, label='Na_se')
    plt.plot(t, Na_di, label='Na_di')
    plt.plot(t, Na_de, label='Na_de')
    plt.title('Sodium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, K_si, label='K_si')
    plt.plot(t, K_se, label='K_se')
    plt.plot(t, K_di, label='K_di')
    plt.plot(t, K_de, label='K_de')
    plt.title('Potassium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, Cl_si, label='Cl_si')
    plt.plot(t, Cl_se, label='Cl_se')
    plt.plot(t, Cl_di, label='Cl_di')
    plt.plot(t, Cl_de, label='Cl_de')
    plt.title('Chloride concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()
