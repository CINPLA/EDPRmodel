#from leakycell import LeakyCell
from pump import Pump
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class PinskyRinzel(Pump):
    """ A two plus two compartment cell model with Pinsky-Rinzel mechanisms and pumps.

    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de):
        Pump.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, k_rest_si, k_rest_se, k_rest_di, k_rest_de)
        self.Ca_si = Ca_si # flytte disse til LeakyCell?
        self_Ca_se = Ca_se
        self.Ca_di = Ca_di
        self_Ca_de = Ca_de


    def alpha_m(phi_sm):
        phi_1 = phi_sm*1e3 + 46.9
        alpha = - 0.32 * phi_1 / (exp(-phi_1 / 4) - 1.)
        return alpha

    def beta_m(phi_sm):
        phi_2 = phi_sm*1e3 + 19.9
        beta = 0.28 * phi_2 / (exp(phi_2 / 5.) - 1.)
        return beta

    def alpha_h(phi_sm):
        alpha = 0.128 * exp((-43. - phi_sm*1e3) / 18.)
        return alpha

    def beta_h(phi_sm):
        phi_5 = phi_sm*1e3 + 20.
        beta = 4. / (1 + exp(-phi_5 / 5.))
        return beta

    def alpha_n(phi_sm):
        phi_3 = phi_sm*1e3 + 24.9
        alpha = - 0.016 * phi_3 / (exp(-phi_3 / 5.) - 1)
        return alpha

    def beta_n(phi_sm):
        phi_4 = phi_sm*1e3 + 40.
        beta = 0.25 * exp(-phi_4 / 40.)
        return beta

    def alpha_s(phi_dm):
        alpha = 1.6 / (1 + exp(-0.072 * (phi_dm*1e3 - 5.)))
        return alpha

    def beta_s(phi_dm):
        phi_6 = phi_dm*1e3 + 8.9
        beta = 0.02 * phi_6 / (exp(phi_6 / 5.) - 1.)
        return beta

    def alpha_c(phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        phi_8 = phi_dm*1e3 + 50.0
        if phi_dm*1e3 <= -10:
            alpha = 0.0527 * exp(phi_8/11.- phi_7/27.)
        else:
            alpha = 2 * exp(-phi_7 / 27.)
        return alpha

    def beta_c(phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        if phi_dm*1e3 <= -10:
            beta = 2. * exp(-phi_7 / 27.) - alpha_c(phi_dm)
        else:
            beta = 0.
        return beta

    def chi(Ca):
        return min(Ca/2.5e-4, 1.0)

