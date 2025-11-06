import legume
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define constants
a_g = 500e-3
r_holes = 120e-3
d_slab = 200e-3
theta_deg = 6.01
theta = np.deg2rad(theta_deg)

# Compute moiré lattice constant
a_moire = a_g / (2 * np.sin(theta / 2))

# --- NORMALIZE all lengths by a_moire ---
a_g /= a_moire
r_holes /= a_moire
d_slab /= a_moire

# Reciprocal lattice vectors of layer 1
G11 = (2*np.pi/(np.sqrt(3)*a_g)) * np.array([np.sqrt(3), 1])
G12 = (2*np.pi/(np.sqrt(3)*a_g)) * np.array([np.sqrt(3), -1])

# Rotation matrix and rotated reciprocal vectors
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
G21 = R @ G11
G22 = R @ G12

# Moiré reciprocal vectors (difference)
g1_moire = G11 - G21
g2_moire = G12 - G22

# Moiré real-space lattice vectors
G_moire = np.column_stack([g1_moire, g2_moire])
A_moire = 2*np.pi * np.linalg.inv(G_moire.T)
A1 = A_moire[:,0]
A2 = A_moire[:,1]

# Import coordinates of holes
df = pd.read_csv("moire_points.csv")
x = np.array(df.x_cent) / a_moire  # normalize positions
y = np.array(df.y_cent) / a_moire

eps_holes = 1.0  # dielectric constant of holes
eps_slab = 3.45**2  # dielectric constant of slab

# Build lattice and photonic crystal
lattice = legume.Lattice(A1, A2)
phc = legume.PhotCryst(lattice, eps_l=1., eps_u=1.)

# Example: single layer (normalized)
phc.add_layer(d=d_slab, eps_b=eps_slab)
for i in range(len(x)):
    circle = legume.Circle(eps=eps_holes, x_cent=x[i], y_cent=y[i], r=r_holes)
    phc.add_shape(circle)

# Guided-mode expansion
gme = legume.GuidedModeExp(phc, gmax=60)

Gamma = [0, 0]
K = (g1_moire + g2_moire) / 3
M = g1_moire / 2
k_points = [Gamma, K, M, Gamma]
N_points = 10
path = lattice.bz_path(k_points, [N_points, N_points, N_points])
path['labels'] = ['G', 'K', 'M', 'G']

gme.run(kpoints=path['kpoints'],
        gmode_inds=[0],
        gmode_step=1e-4,
        verbose=False)

# --- Band Structure Plot ---
fig, ax = plt.subplots(1, figsize=(10, 8))
legume.viz.bands(gme, figsize=(10, 8), k_units=True, Q=True, ax=ax)
ax.set_xticks(path['k_indexes'])
ax.set_xticklabels(path['labels'])
ax.xaxis.grid(True)

a_moire_real = 500e-3 / (2 * np.sin(np.deg2rad(6.01) / 2))  # back to real scale

def norm_to_thz(y):
    return 300 * y / a_moire_real  

def thz_to_norm(y):
    return y * a_moire_real / 300  

secax = ax.secondary_yaxis('right', functions=(norm_to_thz, thz_to_norm))
secax.set_ylabel('Frequency (THz)', fontsize=12)
ax.set_ylabel('Normalized Frequency (ωa/2πc)', fontsize=12)
plt.tight_layout()
plt.savefig('band_structure.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# --- Density of States (DOS) ---
freqs = gme.freqs
all_freqs = freqs.flatten()
num_bins = 300
freq_min, freq_max = all_freqs.min(), all_freqs.max()
bins = np.linspace(freq_min, freq_max, num_bins)
dos, bin_edges = np.histogram(all_freqs, bins=bins, density=False)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dos = dos / (len(freqs) * (bin_edges[1] - bin_edges[0]))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(bin_centers, dos, label="DOS")
ax.set_xlabel("Frequency")
ax.set_ylabel("Density of States")
ax.legend()
plt.tight_layout()
plt.savefig('dos.png', dpi=300, bbox_inches='tight')
plt.close(fig)

