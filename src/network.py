import jax.numpy as jnp
import jax.random as jrandom
from jax import jit


class CoupledNoisyOscillators:
    def __init__(self, config):
        """Initialize parameters for coupled oscillators with noisy boundary conditions."""
        self.L = config["L"]
        self.dx = config["dx"]
        self.dt = config["dt"]
        self.T = config["T"]
        self.N = int(self.L / self.dx)
        self.M = int(self.T / self.dt)

        self.g_inh = config["g_inh"]  # Inhibitory coupling strength
        self.E_inh = config["E_inh"]  # Inhibitory reversal potential
        self.noise_amplitude = config["noise_amplitude"]
        self.key = jrandom.PRNGKey(config["random_seed"])  # Random seed for noise

        # Initialize two oscillatory neurons
        self.V_m1 = jnp.zeros(self.N)
        self.V_m2 = jnp.zeros(self.N)

        # Stimuli for initial excitation
        self.V_m1 = self.V_m1.at[self.N // 3].set(50.0)
        self.V_m2 = self.V_m2.at[2 * self.N // 3].set(50.0)

        self.lambda_factor = self.dt / self.dx ** 2  # Stability parameter for Lax-Friedrichs

    @jit
    def apply_noisy_boundary(self, V_m, key):
        """Applies noisy boundary conditions to each oscillator independently."""
        noise_left = jrandom.normal(key, shape=(1,)) * self.noise_amplitude
        noise_right = jrandom.normal(key, shape=(1,)) * self.noise_amplitude
        V_m = V_m.at[0].set(noise_left)  # Noise at left boundary
        V_m = V_m.at[-1].set(noise_right)  # Noise at right boundary
        return V_m

    @jit
    def inhibition_interaction(self, V_m1, V_m2):
        """Computes mutual inhibition between the two oscillators."""
        I_inh1 = self.g_inh * (V_m2 - self.E_inh)  # V_m2 inhibits V_m1
        I_inh2 = self.g_inh * (V_m1 - self.E_inh)  # V_m1 inhibits V_m2
        return I_inh1, I_inh2

    @jit
    def lax_friedrichs_step(self, V_m, I_inh, key):
        """Computes the next time step using the Lax-Friedrichs scheme with inhibition and noise."""
        # Standard Lax-Friedrichs update
        V_new = 0.5 * (jnp.roll(V_m, -1) + jnp.roll(V_m, 1)) + \
                self.lambda_factor * (jnp.roll(V_m, -1) - 2 * V_m + jnp.roll(V_m, 1))

        # Apply inhibition
        V_new -= I_inh * self.dt

        # Apply noisy boundary conditions
        V_new = self.apply_noisy_boundary(V_new, key)

        return V_new

    def run(self):
        """Runs the simulation for two coupled oscillators with noise."""
        V_m1_hist = jnp.zeros((self.M, self.N))
        V_m2_hist = jnp.zeros((self.M, self.N))

        for i in range(self.M):
            self.key, subkey1, subkey2 = jrandom.split(self.key, 3)  # Generate new noise keys
            I_inh1, I_inh2 = self.inhibition_interaction(self.V_m1, self.V_m2)

            self.V_m1 = self.lax_friedrichs_step(self.V_m1, I_inh1, subkey1)
            self.V_m2 = self.lax_friedrichs_step(self.V_m2, I_inh2, subkey2)

            V_m1_hist = V_m1_hist.at[i, :].set(self.V_m1)
            V_m2_hist = V_m2_hist.at[i, :].set(self.V_m2)

        return V_m1_hist, V_m2_hist
