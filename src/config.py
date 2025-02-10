SIMULATION_CONFIG = {
    "L": 10.0,  # Fiber length (cm)
    "dx": 0.1,  # Spatial step (cm)
    "dt": 0.01,  # Time step (ms)
    "T": 5.0,  # Total simulation time (ms)
    "Cm": 1.0,  # Membrane capacitance (uF/cm^2)
    "Rm": 1.0,  # Membrane resistance (kΩ·cm^2)
    "Ri": 100.0,  # Intracellular resistivity (Ω·cm)
    "E_inh": -70.0,  # Inhibitory reversal potential (mV)
    "g_inh": 0.1,  # Inhibitory conductance strength
    "noise_amplitude": 0.05,  # Strength of the noise at boundaries
    "random_seed": 42,  # Seed for reproducibility
}
