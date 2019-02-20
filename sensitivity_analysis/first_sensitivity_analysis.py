import uncertainpy as un
import chaospy as cp
from leakwrapper import leak_wrap

model = un.Model(run=leak_wrap, labels=["Time (s)", "Membrane potential, soma (V)"])

parameters = {"g_Na": 0.247,
              "g_K": 0.5,
              "g_Cl": 1.0}

parameters = un.Parameters(parameters)
parameters.set_all_distributions(un.uniform(0.2))

UQ = un.UncertaintyQuantification(model, parameters=parameters)
data = UQ.quantify(seed=10)

