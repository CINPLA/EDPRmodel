def somatic_injection_current_K(neuron, dKdt_si, dKdt_se, I_stim):

    dKdt_si += I_stim / (neuron.V_si * neuron.F * neuron.Z_K)
    dKdt_se -= I_stim / (neuron.V_se * neuron.F * neuron.Z_K)
    
    return dKdt_si, dKdt_se
