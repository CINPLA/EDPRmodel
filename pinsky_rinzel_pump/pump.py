from numpy import exp

class Pump(LeakyCell):
    """A two plus two compartment cell model with Na, K and Cl leak currents, and pumps.

    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de):
        LeakyCell.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de)
        self.rho = 
        self.Ukcc2 = 
        self.Unkcc1 =

    def j_pump(self, Na_i, K_e):
        j = self.rho / (1. + exp((25. - Na_i)/3.)) / (1. + exp(3.5 - K_e)) 

    def j_Na_s(self, phi_sm, E_Na_s):
        j = LeakyCell.j_Na_s(self, phi_sm, E_Na_s) + 3*self.j_pump(self.Na_si, self.K_se)
        return j
       
    def j_K_s(self, phi_sm, E_Cl_s):
        j = LeakyCell.j_K_s(self, phi_sm, E_Cl_s) + 2*self.j_pump(self.Na_si, self.K_se)


