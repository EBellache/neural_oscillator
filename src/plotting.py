import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_membrane_potential(V_m_hist, title="Membrane Potential"):
    """Plots the evolution of membrane potential over time with noisy boundary conditions."""
    plt.figure(figsize=(8, 5))
    plt.imshow(V_m_hist.T, aspect='auto', origin='lower', cmap="coolwarm",
               extent=[0, V_m_hist.shape[0] * 0.01, 0, 10])
    plt.colorbar(label="Membrane Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Position along fiber (cm)")
    plt.title(title)
    plt.show()
