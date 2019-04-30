import uncertainpy as un
import chaospy as cp
from leakwrapper import leak_wrap

# Initialize the model and add labeøs
model = un.Model(run=leak_wrap, labels=["Time (s)", "Membrane potential, soma (V)"])

# Define a parameter dictionary
parameters = {"g_Na": 0.247,
              "g_K": 0.5,
              "g_Cl": 1.0}

# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model, parameters=parameters)

# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)

