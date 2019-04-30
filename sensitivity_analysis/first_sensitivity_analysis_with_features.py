import uncertainpy as un
import chaospy as cp
from pinskyrinzelwrapper import pinskyrinzel

# Initialize the model and add labels
model = un.Model(run=pinskyrinzel, labels=["Time (s)", "Membrane potential (V)"], interpolate=True)

# Define a parameter dictionary
parameters = {"g_Na": 300,
              "g_DR": 150}

# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution 
# within a 20 % interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Initialize the features
features = un.SpikingFeatures(features_to_run="all", threshold=-0.03, end_threshold=-0.01)

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model, parameters=parameters, features=features)

# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)

