from .pump import Pump
import numpy as np

class EDPRmodel(Pump):
    """ Electrodiffusive Pinsky-Rinzel model.

    Attributes
    ----------
    Pump (Class)

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha, \
        Ca0_si, Ca0_di, n, h, s, c, q, z)
    alpha_m(phi_sm), beta_m(phi_sm), alpha_h(phi_sm), beta_h(phi_sm),
    alpha_n(phi_sm), beta_n(phi_sm), alpha_s(phi_dm), beta_s(phi_dm),
    alpha_c(phi_dm), beta_c(phi_dm), chi(), alpha_q(), beta_q(),
    m_inf(phi_sm), z_inf(phi_dm): compute the rate coefficients of the gating particles
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Ca_s(phi_sm, E_Ca_s): compute the Ca2+ flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    j_Ca_d(phi_dm, E_Ca_d): compute the Ca2+ flux across the dendritic membrane
    dkdt(): calculate dk/dt for all ion species k
    dmdt(): calculate dm/dt for all gating particles m
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha, Ca0_si, Ca0_di, n, h, s, c, q, z):

        Pump.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha)

        self.Ca0_si = Ca0_si # basal Ca concentration
        self.Ca0_di = Ca0_di
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q
        self.z = z

        # conductances [S * m**-2]
        self.g_Na = 300.
        self.g_DR = 150.
        self.g_Ca = 118.
        self.g_AHP = 8.
        self.g_C = 150.

        # Calcium decay rate [s**-1]
        self.U_Cadec = 75.

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
        alpha = 1.6 / (1 + np.exp(-0.072 * (phi_dm*1000 - 5.)))
        alpha = alpha*1000
        return alpha

    def beta_s(self, phi_dm):
        phi_6 = phi_dm*1000 + 8.9
        beta = 0.02 * phi_6 / (np.exp(phi_6 / 5.) - 1.)
        beta = beta*1000
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
        return min((self.free_Ca_di-99.8e-6)/2.5e-4, 1.0)

    def alpha_q(self):
        return min(2e4*(self.free_Ca_di-99.8e-6), 10.0) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_sm):
        return self.alpha_m(phi_sm) / (self.alpha_m(phi_sm) + self.beta_m(phi_sm))

    def z_inf(self, phi_dm):
        phi_half = -30
        k = 1
        return 1/(1 + np.exp((phi_dm*1000 - phi_half)/k))

    def j_Na_s(self, phi_sm, E_Na_s):
        j = Pump.j_Na_s(self, phi_sm, E_Na_s) \
            + self.g_Na * self.m_inf(phi_sm)**2 * self.h * (phi_sm - E_Na_s) / (self.F*self.Z_Na)\
            - 2*self.U_Cadec * (self.Ca_si - self.Ca0_si)*self.V_si/self.A_s
        return j

    def j_K_s(self, phi_sm, E_K_s):
        j = Pump.j_K_s(self, phi_sm, E_K_s) \
            + self.g_DR * self.n * (phi_sm - E_K_s) / (self.F*self.Z_K)
        return j
    
    def j_Ca_s(self):
        j =  self.U_Cadec * (self.Ca_si - self.Ca0_si)*self.V_si/self.A_s
        return j
    
    def j_Na_d(self, phi_dm, E_Na_d):
        j = Pump.j_Na_d(self, phi_dm, E_Na_d) \
            - 2*self.U_Cadec * (self.Ca_di - self.Ca0_di)*self.V_di/self.A_d
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = Pump.j_K_d(self, phi_dm, E_K_d) \
            + self.g_AHP * self.q * (phi_dm - E_K_d) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_dm - E_K_d) / (self.F*self.Z_K)
        return j

    def j_Ca_d(self, phi_dm, E_Ca_d):
        j = self.g_Ca * self.s**2 * self.z * (phi_dm - E_Ca_d) / (self.F*self.Z_Ca) \
            + self.U_Cadec * (self.Ca_di - self.Ca0_di)*self.V_di/self.A_d
        return j

    def dkdt(self):

        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = Pump.membrane_potentials(self)
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = Pump.reversal_potentials(self)
        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dXdt_si, dXdt_se, dXdt_di, dXdt_de = Pump.dkdt(self)

        j_Ca_sm = self.j_Ca_s()
        j_Ca_dm = self.j_Ca_d(phi_dm, E_Ca_d)

        dCadt_si = dCadt_si - j_Ca_sm*(self.A_s / self.V_si)
        dCadt_se = dCadt_se + j_Ca_sm*(self.A_s / self.V_se)
        dCadt_di = dCadt_di - j_Ca_dm*(self.A_d / self.V_di)
        dCadt_de = dCadt_de + j_Ca_dm*(self.A_d / self.V_de)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dXdt_si, dXdt_se, dXdt_di, dXdt_de

    def dmdt(self):
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = Pump.membrane_potentials(self)
        
        dndt = self.alpha_n(phi_sm)*(1.0-self.n) - self.beta_n(phi_sm)*self.n
        dhdt = self.alpha_h(phi_sm)*(1.0-self.h) - self.beta_h(phi_sm)*self.h 
        dsdt = self.alpha_s(phi_dm)*(1.0-self.s) - self.beta_s(phi_dm)*self.s
        dcdt = self.alpha_c(phi_dm)*(1.0-self.c) - self.beta_c(phi_dm)*self.c
        dqdt = self.alpha_q()*(1.0-self.q) - self.beta_q()*self.q
        dzdt = (self.z_inf(phi_dm) - self.z)
        
        return dndt, dhdt, dsdt, dcdt, dqdt, dzdt
