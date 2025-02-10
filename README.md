# Classical Neural Oscillator (JAX-Based)

## 📌 Project Overview
This project simulates **noisy coupled neural oscillators** using **JAX**. It models **impulse propagation** with **Lax-Friedrichs stability**, **mutual inhibition**, and **noisy boundary conditions**, leading to **stochastic resonance**.

---

## 📖 Scientific Principles

### **1️⃣ Noisy Boundary Conditions for Stochastic Oscillations**
To introduce **random neural fluctuations**, we apply **noisy boundary conditions**:

$$
V_m(0, t) = \eta(t), \quad V_m(L, t) = \eta(t)
$$

where $\eta(t)$ is Gaussian noise. This **models randomness** in neural circuits and generates **chaotic oscillations**.

---

### **2️⃣ Coupled Oscillators with Mutual Inhibition**
Two oscillators are **inhibitorily coupled**, following:

$$
I_{\text{inh}} = g_{\text{inh}} (V_m - E_{\text{inh}})
$$

which causes **alternating waveforms**, mimicking **biological rhythms**.

---

## 🚀 Getting Started
### **📦 Installation**
```sh
pip install -r requirements.txt
