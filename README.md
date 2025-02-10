# Classical Neural Oscillator (JAX-Based)

## 📌 Project Overview
This project simulates **coupled neural oscillators** using **JAX** to model **impulse propagation** and **inhibition-based oscillations**. The simulation is based on:
- The **cable equation** for membrane voltage dynamics.
- The **Lax-Friedrichs scheme** for stable numerical solution.
- **Mutual inhibition** to generate oscillatory behavior.

The goal is to create **classical oscillatory neural circuits**, useful for understanding **central pattern generators (CPGs)**, **neural rhythm generation**, and **brainwave activity**.

---

## 📖 Scientific Principles

### **1️⃣ Membrane Potential and the Cable Equation**
Impulse propagation in a **neural fiber** follows the **cable equation**, derived from the core-conductor model:

$$
C_m \frac{\partial V_m}{\partial t} = \frac{1}{R_m} \frac{\partial^2 V_m}{\partial x^2} - I_{\text{ion}}
$$

where:
- $C_m$ is the **membrane capacitance** (μF/cm²),
- $R_m$ is the **membrane resistance** (kΩ·cm²),
- $V_m(x,t)$ is the **membrane potential** at position $x$ and time $t$,
- $I_{\text{ion}}$ is the **ionic current**.

Using **finite differences**, the discretized form is:

$$
\frac{V_{m,j}^{i+1} - V_{m,j}^{i}}{\Delta t} = \frac{1}{C_m} \left( \frac{V_{m,j+1}^{i} - 2V_{m,j}^{i} + V_{m,j-1}^{i}}{\Delta x^2} - \frac{V_{m,j}^{i}}{R_m} \right)
$$

This allows numerical simulation of **action potential propagation**.

---

### **2️⃣ Lax-Friedrichs Modification for Stability**
To improve stability, we implement the **Lax-Friedrichs scheme**, which modifies the finite difference update by adding a **time-averaging term**:

$$
V_{m,j}^{i+1} = \frac{1}{2} \left( V_{m,j-1}^{i} + V_{m,j+1}^{i} \right) + \lambda \left( V_{m,j+1}^{i} - 2V_{m,j}^{i} + V_{m,j-1}^{i} \right)
$$

where:
- $\lambda = \frac{\Delta t}{\Delta x^2}$ is the **stability parameter**.
- The **first term** stabilizes the numerical solution.
- The **second term** represents the **cable equation dynamics**.

This method **reduces oscillations** and ensures the numerical solver remains stable, even for larger **time steps**.

---

### **3️⃣ Neural Oscillations via Inhibition**
To generate **oscillatory behavior**, we introduce **mutual inhibition**:

$$
I_{\text{inh}} = g_{\text{inh}} (V_m - E_{\text{inh}})
$$

where:
- $g_{\text{inh}}$ is the **inhibitory conductance** (time-dependent).
- $E_{\text{inh}}$ is typically **-70 mV** (hyperpolarizing effect).
- This **mimics synaptic inhibition** in neurons.

By coupling two neurons with **reciprocal inhibition**, they can **oscillate** in a pattern seen in **biological networks like central pattern generators (CPGs)**.

---

## 🚀 Getting Started

### **📦 Installation**
Ensure you have **Python 3.8+** installed. Then install dependencies:

```sh
pip install -r requirements.txt
