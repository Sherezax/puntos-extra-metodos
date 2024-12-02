import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

L = 1
tf = 0.2
Nt = 500
Nx = 500

X, Y = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, 0.01, 2))
t = np.linspace(0, tf, Nt)

@njit
def U(x, t):
    return 30 * np.exp(-4 * np.power(np.pi, 2) * t) * np.sin(2 * np.pi * x)

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.patch.set_facecolor('black')
ax.set_facecolor('#2c2c2c')


distr = ax.pcolormesh(X, Y, U(X, 0), cmap='plasma', shading="nearest", vmin=-30, vmax=30)

cbar = plt.colorbar(distr, ax=ax, label='Temperatura en °C')
cbar.set_ticks(np.linspace(-30, 30, 5))
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.set_yticklabels([f'{int(tick)} °C' for tick in np.linspace(-30, 30, 5)], color='white')
cbar.set_label('Temperatura en °C', color='white')

ax.set_xlabel('Posición', color='white')
ax.set_ylabel('Altura', color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.grid(color='white', linestyle='--', linewidth=0.5)


title = ax.set_title(f"Evolución Temporal: t = 0.000 s", color='white')

def update(frame):
    distr.set_array(U(X, t[frame]).flatten())
    title.set_text(f"Evolución Temporal: t = {t[frame]:.3f} s")
    return distr, title

plt.xlim([X.min() - 0.1, X.max() + 0.1])
plt.ylim([Y.min() - 0.03, Y.max() + 0.03])

ani = animation.FuncAnimation(fig=fig, func=update, frames=Nt, interval=20)


ani.save('Animación puntos extra.gif', dpi=100, writer='pillow')
plt.show()
