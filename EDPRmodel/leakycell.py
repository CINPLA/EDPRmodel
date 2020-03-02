import numpy as np
import warnings
warnings.filterwarnings("error")

class LeakyCell(): 
    """A two plus two compartment neuron model with Na+, K+, and Cl- leak currents.

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha)
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    j_k_diff(D_k, tortuosity, k_s, k_d): compute the axial diffusion flux of ion k
    j_k_drift(D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d): compute the axial drift flux of ion k
    conductivity_k(D_k, Z_k, tortuosity, k_s, k_d): compute axial conductivity of ion k
    total_charge(k, V): calculate the total charge within volume V
    nernst_potential(Z, k_i, k_e): calculate the reversal potential of ion k
    reversal_potentials(): calculate the reversal potentials of all ion species
    membrane_potentials(): calculate the membrane potentials
    dkdt(): calculate dk/dt for all ion species k
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha):
        
        # absolute temperature [K]
        self.T = T

        # ion concentraions [mol * m**-3]
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
        self.Ca_si = Ca_si
        self.Ca_se = Ca_se 
        self.Ca_di = Ca_di 
        self.Ca_de = Ca_de
        self.free_Ca_si = 0.01*Ca_si
        self.free_Ca_di = 0.01*Ca_di
        self.X_si = X_si
        self.X_se = X_se
        self.X_di = X_di
        self.X_de = X_de

        # membrane capacitance [F * m**-2]
        self.C_sm = 3e-2 # Pinsky and Rinzel 1994
        self.C_dm = 3e-2 # Pinsky and Rinzel 1994
       
        # volumes and areas
        self.alpha = alpha
        self.A_s = 616e-12             # [m**2]
        self.A_d = 616e-12             # [m**2]
        self.A_i = self.alpha*self.A_s # [m**2]
        self.A_e = self.A_i/2.         # [m**2]
        self.V_si = 1437e-18           # [m**3]
        self.V_di = 1437e-18           # [m**3]
        self.V_se = 718.5e-18          # [m**3]
        self.V_de = 718.5e-18          # [m**3]
        self.dx = 667e-6               # [m]

        # diffusion constants [m**2 s**-1]
        self.D_Na = 1.33e-9 # Halnes et al. 2013
        self.D_K = 1.96e-9  # Halnes et al. 2013 
        self.D_Cl = 2.03e-9 # Halnes et al. 2013
        self.D_Ca = 0.71e-9 # Halnes et al. 2016

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.
        self.Z_Ca = 2.
        self.Z_X = -1.

        # constants
        self.F = 9.648e4    # [C * mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        # conductances [S * m**-2]
        self.g_Na_leak = 0.247 # Wei et al. 2014
        self.g_K_leak = 0.5    # Wei et al. 2014
        self.g_Cl_leak = 1.0   # Wei et al. 2014

    def j_Na_s(self, phi_sm, E_Na_s):
        j = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na)
        return j 

    def j_K_s(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl)
        return j

    def j_Na_d(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) 
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) 
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) 
        return j

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
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca, self.Z_X]
        q = 0.0
        for i in range(0, 5):
            q += Z_k[i]*k[i]
        q = q*self.F*V
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
        E_Ca_s = self.nernst_potential(self.Z_Ca, self.free_Ca_si, self.Ca_se)
        E_Ca_d = self.nernst_potential(self.Z_Ca, self.free_Ca_di, self.Ca_de)
        return E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d

    def membrane_potentials(self):
        I_i_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de)

        q_di = self.total_charge([self.Na_di, self.K_di, self.Cl_di, self.Ca_di, self.X_di], self.V_di)
        q_si = self.total_charge([self.Na_si, self.K_si, self.Cl_si, self.Ca_si, self.X_si], self.V_si)

        phi_di = q_di / (self.C_dm * self.A_d)
        phi_se = (phi_di - self.dx * I_i_diff / sigma_i - self.A_e * self.dx * I_e_diff / (self.A_i * sigma_i) - q_si / (self.C_sm * self.A_s)) / (1 + self.A_e*sigma_e/(self.A_i*sigma_i))
        phi_si = q_si / (self.C_sm * self.A_s) + phi_se
        phi_de = 0.
        phi_sm = phi_si - phi_se
        phi_dm = phi_di - phi_de

        return phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm

    def dkdt(self):
       
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = self.membrane_potentials()
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = self.reversal_potentials()

        j_Na_sm = self.j_Na_s(phi_sm, E_Na_s)
        j_K_sm = self.j_K_s(phi_sm, E_K_s)
        j_Cl_sm = self.j_Cl_s(phi_sm, E_Cl_s)

        j_Na_dm = self.j_Na_d(phi_dm, E_Na_d)
        j_K_dm = self.j_K_d(phi_dm, E_K_d)    
        j_Cl_dm = self.j_Cl_d(phi_dm, E_Cl_d)

        j_Na_i = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di, phi_si, phi_di) 
        j_K_i = self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di, phi_si, phi_di)
        j_Cl_i = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di, phi_si, phi_di)
        j_Ca_i = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di, phi_si, phi_di)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de, phi_se, phi_de)

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

        dCadt_si = - j_Ca_i*(self.A_i / self.V_si)
        dCadt_di = j_Ca_i*(self.A_i / self.V_di)
        dCadt_se = - j_Ca_e*(self.A_e / self.V_se)
        dCadt_de = j_Ca_e*(self.A_e / self.V_de)

        dXdt_si = 0
        dXdt_di = 0
        dXdt_se = 0
        dXdt_de = 0

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dXdt_si, dXdt_se, dXdt_di, dXdt_de
