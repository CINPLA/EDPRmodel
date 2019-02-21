def somatic_injection_current_res(neuron, dresdt_si, dresdt_se, I_stim):

    Z = 1.0
    dresdt_si += I_stim / (neuron.V_si * neuron.F * Z)
    dresdt_se -= I_stim / (neuron.V_se * neuron.F * Z)
    
    return dresdt_si, dresdt_se
