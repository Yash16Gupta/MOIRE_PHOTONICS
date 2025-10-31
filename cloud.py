import legume
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- Define structure parameters ----
a_g = 500e-3
r_holes = 120e-3
eps_holes = 1.0
n_slab = 3.45
D_slab = 200e-3
theta = np.deg2rad(6.01)

# ---- Reciprocal lattice vectors ----
G11 = (2*np.pi/(np.sqrt(3)*a_g)) * np.array([np.sqrt(3), 1])
G12 = (2*np.pi/(np.sqrt(3)*a_g)) * np.array([np.sqrt(3), -1])

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
G21 = R @ G11
G22 = R @ G12

g1_moire = G11 - G21
g2_moire = G12 - G22
G_moire = np.column_stack([g1_moire, g2_moire])
A_moire = 2*np.pi * np.linalg.inv(G_moire.T)
A1 = A_moire[:,0]
A2 = A_moire[:,1]

# ---- Load points ----
df = pd.read_csv("moire_points.csv")
x = np.array(df.x_cent)
y = np.array(df.y_cent)

# ---- Photonic crystal setup ----
lattice = legume.Lattice(A1, A2)
phc = legume.PhotCryst(lattice, eps_l=1., eps_u=1.)
phc.add_layer(d=D_slab, eps_b=n_slab**2)

for i in range(len(x)):
    circle = legume.Circle(eps=eps_holes, x_cent=x[i], y_cent=y[i], r=r_holes)
    phc.add_shape(circle)

# ---- Guided mode expansion ----
N_Res = 100
gme = legume.GuidedModeExp(phc, gmax=6)
Gamma = [0,0]
K = (g1_moire + g2_moire)/3
M = g1_moire/2
k_points = [Gamma, K, M, Gamma]
N_points = 10
path = lattice.bz_path(k_points, [N_points,N_points,N_points])
path['labels'] = ['G', 'K', 'M', 'G']

gme.run(kpoints=path['kpoints'],
        gmode_inds=[0],
        numeig=10,
        eig_solver='eigsh',
        eig_sigma=3.2,
        verbose=False)

# ---- 1️⃣ Band structure plot ----
fig, ax = plt.subplots(1, figsize=(15, 10))
legume.viz.bands(gme, figsize=(10, 10), k_units=True, Q=True, ax=ax)
ax.set_xticks(path['k_indexes'])
ax.set_xticklabels(path['labels'])
ax.xaxis.grid(True)

a_moire = a_g / (2 * np.sin(np.deg2rad(6.01) / 2))

def norm_to_thz(y):
    return 300 * y / a_moire  

def thz_to_norm(y):
    return y * a_moire / 300  

secax = ax.secondary_yaxis('right', functions=(norm_to_thz, thz_to_norm))
secax.set_ylabel('Frequency (THz)', fontsize=12)
ax.set_ylabel('Normalized Frequency (ωa/2πc)', fontsize=12)

plt.tight_layout()
plt.savefig('band_structure.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---- 2️⃣ Density of States (DOS) plot ----
freqs = gme.freqs
all_freqs = freqs.flatten()
num_bins = 300
freq_min, freq_max = all_freqs.min(), all_freqs.max()
bins = np.linspace(freq_min, freq_max, num_bins)
dos, bin_edges = np.histogram(all_freqs, bins=bins, density=False)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dos = dos / (len(freqs) * (bin_edges[1] - bin_edges[0]))

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(bin_centers, dos, label="DOS")
ax2.set_xlabel("Frequency")
ax2.set_ylabel("Density of States")
ax2.legend()
plt.tight_layout()
plt.savefig('density_of_states.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# ---- 3️⃣ Frequency dispersion plot ----
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(gme.freqs)
ax3.set_xlabel("k-point index")
ax3.set_ylabel("Frequency (normalized)")
ax3.set_title("Frequency dispersion across k-points")
plt.tight_layout()
plt.savefig('frequency_dispersion.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# ---- Summary ----
print("✅ Saved plots:")
print(" - band_structure.png")
print(" - density_of_states.png")
print(" - frequency_dispersion.png")
print(f"📂 Directory: {os.getcwd()}")
