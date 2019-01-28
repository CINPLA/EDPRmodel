#from leakycell import LeakyCell
from pump import Pump
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class PinskyRinzel(Pump):
    """ A two plus two compartment cell model with Pinsky-Rinzel mechanisms and pumps.

    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de, n, h, s, c, q):
        Pump.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de)
        self.Ca_si = Ca_si # flytte disse til LeakyCell?
        self.Ca_se = Ca_se
        self.Ca_di = Ca_di
        self.Ca_de = Ca_de
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
        alpha = - 0.32 * phi_1 / (exp(-phi_1 / 4) - 1.)
        return alpha

    def beta_m(self, phi_sm):
        phi_2 = phi_sm*1e3 + 19.9
        beta = 0.28 * phi_2 / (exp(phi_2 / 5.) - 1.)
        return beta

    def alpha_h(self, phi_sm):
        alpha = 0.128 * exp((-43. - phi_sm*1e3) / 18.)
        return alpha

    def beta_h(self, phi_sm):
        phi_5 = phi_sm*1e3 + 20.
        beta = 4. / (1 + exp(-phi_5 / 5.))
        return beta

    def alpha_n(self, phi_sm):
        phi_3 = phi_sm*1e3 + 24.9
        alpha = - 0.016 * phi_3 / (exp(-phi_3 / 5.) - 1)
        return alpha

    def beta_n(self, phi_sm):
        phi_4 = phi_sm*1e3 + 40.
        beta = 0.25 * exp(-phi_4 / 40.)
        return beta

    def alpha_s(self, phi_dm):
        alpha = 1.6 / (1 + exp(-0.072 * (phi_dm*1e3 - 5.)))
        return alpha

    def beta_s(self, phi_dm):
        phi_6 = phi_dm*1e3 + 8.9
        beta = 0.02 * phi_6 / (exp(phi_6 / 5.) - 1.)
        return beta

    def alpha_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        phi_8 = phi_dm*1e3 + 50.0
        if phi_dm*1e3 <= -10:
            alpha = 0.0527 * exp(phi_8/11.- phi_7/27.)
        else:
            alpha = 2 * exp(-phi_7 / 27.)
        return alpha

    def beta_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        if phi_dm*1e3 <= -10:
            beta = 2. * exp(-phi_7 / 27.) - alpha_c(phi_dm)
        else:
            beta = 0.
        return beta

    def chi(self, Ca):
        return min(Ca/2.5e-4, 1.0)

    def alpha_q(self, Ca):
        return min(20*Ca, 0.01)

    def beta_q(self):
        return 0.001

    def m_inf(self, phi_sm):
        return self.alpha_m(phi_sm) / (self.alpha_m(phi_sm) + self.beta_m(phi_sm))

    def j_Na_s(self, phi_sm, E_Na_s):
        j = Pump.j_Na_s(self, phi_sm, E_Na_s) \
            + self.g_Na * m_inf(phi_sm)**2 * self.h * (phi_sm - E_Na_s)

    def j_K_s(self, phi_sm, E_K_s):
        j = Pump.j_K_s(self, phi_sm, E_K_s) \
            + self.g_DR * self.n * (phi_sm - E_K_s)

