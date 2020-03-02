from .leakycell import LeakyCell
import numpy as np

class Pump(LeakyCell):
    """A two plus two compartment neuron model with Na, K and Cl leak currents and pumps.

    Attributes
    ----------
    LeakyCell (Class)

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha)
    j_pump(Na_i, K_e): compute the Na+/K+ pump flux across given membrane
    j_kcc2(K_i, K_e, Cl_i, Cl_e): compute the K+/Cl- co-transporter flux across given membrane
    j_nkcc1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e): compute the Na+/K+/Cl- co-transporter flux across given membrane
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha):
        LeakyCell.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha)
        self.rho = 1.87e-6      # Wei et al. 2014
        self.U_kcc2 = 7.00e-7   # Wei et al. 2014
        self.U_nkcc1 = 2.33e-7  # Wei et al. 2014

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
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
       
    def j_K_s(self, phi_sm, E_K_s):
        j = LeakyCell.j_K_s(self, phi_sm, E_K_s) \
            - 2*self.j_pump(self.Na_si, self.K_se) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = LeakyCell.j_Cl_s(self, phi_sm, E_Cl_s) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + 2*self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
            
    def j_Na_d(self, phi_dm, E_Na_d):
        j = LeakyCell.j_Na_d(self, phi_dm, E_Na_d) \
            + 3*self.j_pump(self.Na_di, self.K_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = LeakyCell.j_K_d(self, phi_dm, E_K_d) \
            - 2*self.j_pump(self.Na_di, self.K_de) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = LeakyCell.j_Cl_d(self, phi_dm, E_Cl_d) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + 2*self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

