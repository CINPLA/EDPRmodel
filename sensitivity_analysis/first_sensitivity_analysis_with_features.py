import uncertainpy as un
import chaospy as cp
from pinskyrinzelwrapper import pinskyrinzel

model = un.Model(run=pinskyrinzel, labels=["Time (s)", "Membrane potential (V)"], interpolate=True)

parameters = {"I_stim": 1500e-12,
              "g_Na": 300.,
              "g_DR": 150.}

parameters = un.Parameters(parameters)
parameters.set_all_distributions(un.uniform(0.05))

features = un.SpikingFeatures(features_to_run="all", threshold=-0.03, end_threshold=-0.01)

UQ = un.UncertaintyQuantification(model, parameters=parameters, features=features)
data = UQ.quantify(seed=10)
