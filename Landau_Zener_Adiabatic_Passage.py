import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Detuning range
delta_min = -20
delta_max = 20

#Chirp rates: alpha/Omega^2
chirp_rates = [0.2, 2, 20]

#Data points for detuning
delta_values = np.linspace(delta_min, delta_max, 2000)

plt.figure(figsize=(10,6))

#Chirp
for beta in chirp_rates:
    
    #Dimensionless time
    tau_min = delta_min/beta
    tau_max = delta_max/beta
    tau_values = delta_values/beta

    #Dimensionless Hamiltonian
    def hamiltonian(tau):
        delta = beta*tau

        return 0.5*np.array([[delta, 1], [1, -delta]],dtype=complex)

    #Schrodinger equation
    def schrodinger(tau, psi):
        H = hamiltonian(tau)
        return -1j*H @ psi

    #initial detuning
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian(tau_min))

    psi_initial = eigenvectors[:,0]

    #integration
    solution = solve_ivp(
        schrodinger, (tau_min, tau_max), psi_initial,
        t_eval= tau_values, rtol = 1e-9, atol = 1e-11
    )

    #excited state population
    P_e = np.abs(solution.y[1])**2

    plt.plot(
        delta_values, P_e,
        label=rf"$\alpha/\Omega^2={beta}$"
    )


plt.xlabel(r"$\Delta/\Omega$", fontsize=14)
plt.ylabel(r"$P_e$", fontsize=14)
plt.title("Landau_Zener transition")
plt.xlim(delta_min, delta_max)
plt.ylim(-0.02, 1.05)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()