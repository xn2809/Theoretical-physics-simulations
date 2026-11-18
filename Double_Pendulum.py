import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#parameters
g = 9.81    #m/S^2
m1, m2 = 1.0, 1.0   #kg
L1, L2 = 10.0, 1.0   #m

#angles and velocities
theta1_0 = np.radians(120)
theta2_0 = np.radians(-10)
omega1_0 = 0.0
omega2_0 = 0.0

dt = 0.001
steps_per_frame = 10

t_window = 20.0

trail_length = 250

#physical calculations
def derivative(state):
    th1, w1, th2, w2 = state
    d = th1 - th2
    den = 2*m1 + m2 - m2*np.cos(2*d)

    a1 = (-g*(2*m1+m2)*np.sin(th1)) - m2*g*np.sin(th1-2*th2) - 2*np.sin(d)*m2*((w2**2*L2+w1**2*L1*np.cos(d)))/(L1*den)

    a2 = (2*np.sin(d)*(w1**2*L1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*L2*m2*np.cos(d)))/(L2*den)

    return np.array([w1, a1, w2, a2])

def rk4(state, h):
    k1=derivative(state)
    k2=derivative(state+0.5*h*k1)
    k3=derivative(state+0.5*h*k2)
    k4=derivative(state+h*k3)

    return state + (h/6)*(k1+2*k2+2*k3+k4)

def energies(state):
    th1, w1, th2, w2 = state
    ke = (0.5*m1*(L1*w1)**2+0.5*m2*((L1*21)**2+(L2*w2)+2*L1*L2*w1*w2*np.cos(th1-th2)))

    pe = g*((m1+m2)*L1*(1-np.cos(th1))+m2*L2*(1-np.cos(th2)))

    return ke, pe

def position(state):
    th1, _, th2, _ = state
    x1, y1 = L1*np.sin(th1), -L1*np.cos(th1)

    x2, y2 = x1 + L2*np.sin(th2), y1 - L2*np.cos(th2)

    return x1, y1, x2, y2


#plot - simulation
state = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=float)
t = 0.0

times, KE, PE, TE = [], [], [], []
trail_x, trail_y= [], []

fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Double pendulum")

#Pendulum panel
R = (L1 + L2) * 1.1
ax_p.set_xlim(-R, R)
ax_p.set_ylim(-R, R)
ax_p.set_aspect("equal")
ax_p.grid(alpha=0.3)
ax_p.set_title("Motion")
rods, = ax_p.plot([], [], "o-", color="black", lw=2, ms=9)
trail, = ax_p.plot([], [], "-", color="crimson", lw=1, alpha=0.6)
time_text = ax_p.text(0.02, 0.96, "", transform=ax_p.transAxes, va="top")

# Energy panel
E_max = (m1 + m2) * g * L1 * 2 + m2 * g * L2 * 2   # loose upper bound for the y-axis
ax_e.set_xlim(0, t_window if t_window else 20)
ax_e.set_ylim(0, 1.05 * (m1 + m2) * g * (2 * L1 + 2 * L2) / 2 * 1.5)
ax_e.set_xlabel("time (s)")
ax_e.set_ylabel("energy (J)")
ax_e.set_title("Energy")
ax_e.grid(alpha=0.3)
line_ke, = ax_e.plot([], [], label="Kinetic", color="skyblue")
line_pe, = ax_e.plot([], [], label="Potential", color="pink")
line_te, = ax_e.plot([], [], label="Total", color="black", ls="--")
ax_e.legend(loc="upper right")


def update(_frame):
    global state, t

    for _ in range(steps_per_frame):
        state = rk4(state, dt)
        t += dt

    ke, pe = energies(state)
    times.append(t)
    KE.append(ke)
    PE.append(pe)
    TE.append(ke + pe)

    # Pendulum drawing
    x1, y1, x2, y2 = position(state)
    rods.set_data([0, x1, x2], [0, y1, y2])
    trail_x.append(x2)
    trail_y.append(y2)
    trail.set_data(trail_x[-trail_length:], trail_y[-trail_length:])
    time_text.set_text(f"t = {t:5.2f} s")

    # Energy drawing
    line_ke.set_data(times, KE)
    line_pe.set_data(times, PE)
    line_te.set_data(times, TE)

    if t_window and t > t_window:           # scroll the x-axis
        ax_e.set_xlim(t - t_window, t)
    elif not t_window:
        ax_e.set_xlim(0, max(20, t))

    top = max(TE) * 1.2
    ax_e.set_ylim(0, top)
    return rods, trail, time_text, line_ke, line_pe, line_te


anim = FuncAnimation(fig, update, interval=16, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
