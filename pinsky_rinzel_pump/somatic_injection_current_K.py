def somatic_injection_current_K(neuron, I_stim):

    neuron.dKdt_si += I_stim / (neuron.V_si * neuron.F * neuron.Z_K)
    neuron.dKdt_se -= I_stim / (neuron.V_se * neuron.F * neuron.Z_K)
