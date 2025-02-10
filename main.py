from src.network import CoupledNoisyOscillators
from src.plotting import plot_membrane_potential
from src.config import SIMULATION_CONFIG

# Initialize coupled oscillators with noisy boundary conditions
oscillators = CoupledNoisyOscillators(SIMULATION_CONFIG)
V_m1_hist, V_m2_hist = oscillators.run()

# Plot results for both oscillators
plot_membrane_potential(V_m1_hist, title="Oscillator 1 (with Noise & Inhibition)")
plot_membrane_potential(V_m2_hist, title="Oscillator 2 (with Noise & Inhibition)")
