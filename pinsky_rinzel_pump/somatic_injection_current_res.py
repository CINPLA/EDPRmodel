def somatic_injection_current_res(neuron, I_stim):

    Z = 1.0
    neuron.dresdt_si += I_stim / (neuron.V_si * neuron.F * Z)
    neuron.dresdt_se -= I_stim / (neuron.V_se * neuron.F * Z)
