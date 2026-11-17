import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Landau-Zener parameters
beta = 20

tau_min = -20 / beta
tau_max = 20 / beta

tau_values = np.linspace(tau_min, tau_max, 2000)

#hamiltonian
def hamiltonian(tau):
    delta = beta * tau

    return 0.5 * np.array([
        [delta, 1],
        [1, -delta]
    ], dtype=complex)

#schrodinger equation
def schrodinger(tau, psi):
    H = hamiltonian(tau)
    return -1j * H @ psi

#ground state
eigenvalues, eigenvectors = np.linalg.eigh(
    hamiltonian(tau_min)
)

psi_initial = eigenvectors[:, 0]

#integration
solution = solve_ivp(
    schrodinger,
    (tau_min, tau_max),
    psi_initial,
    t_eval=tau_values,
    rtol=1e-9,
    atol=1e-11
)

#probability amplitudes
c0 = solution.y[0]
c1 = solution.y[1]

#Bloch coordinates
x_bloch = 2 * np.real(np.conj(c0) * c1)
y_bloch = 2 * np.imag(np.conj(c0) * c1)
z_bloch = np.abs(c0)**2 - np.abs(c1)**2

#spherical coordinates
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)

theta, phi = np.meshgrid(theta, phi)

#cartesian conversion
x = np.sin(theta)*np.cos(phi)
y = np.sin(phi)*np.sin(theta)
z = np.cos(theta)

#figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

#sphere
ax.plot_surface(
    x,y,z,
    alpha = 0.15,
    edgecolor = "pink",
    linewidth = 0.3
)

#quantum state
ax.plot(
    x_bloch, y_bloch, z_bloch,
    linewidth=2, label = "Quantum state" 
)

#initial and final states
ax.scatter(
    x_bloch[0], y_bloch[0], z_bloch[0],
    color='green', s=60, label='Initial state'
)

ax.scatter(
    x_bloch[-1], y_bloch[-1], z_bloch[-1],
    color='red', s=60, label='Final state'
)

#axes
ax.quiver(0, 0, 0, 1, 0, 0, color="skyblue", label="X")
ax.quiver(0, 0, 0, 0, 1, 0, color="green", label="Y")
ax.quiver(0, 0, 0, 0, 0, 1, color="crimson", label="Z")

#labels
ax.text(1.1, 0, 0, "X")
ax.text(0, 1.1, 0, "Y")
ax.text(0, 0, 1.1, "|0>")
ax.text(0, 0, -1.15, "|1>")

ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])

ax.set_box_aspect([1, 1, 1])
ax.set_title("Qubit")

plt.show()
