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
        self.Ca_si = Ca_si
        self_Ca_se = Ca_se
        self.Ca_di = Ca_di
        self_Ca_de = Ca_de

