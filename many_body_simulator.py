"""
P452 Project 2 — Many-Body Quantum Simulator
=============================================
Streamlit app covering:
  • Phase 1 : Exact diagonalization of the spin-1/2 Heisenberg model
              on 1D chains, 2D square lattices, and the 3-site triangle.
              Uses Sz_total block-diagonalization + Lanczos for large N.
  • Phase 2 : Bose-Fermi mixture in a spherical harmonic trap, solved
              self-consistently with Thomas-Fermi (bosons) + Local-Density
              Approximation (fermions).
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from math import comb as _comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import brentq

import streamlit as st


# ============================================================================
# ============   PHASE 1 :  HEISENBERG MODEL EXACT DIAGONALIZATION   =========
# ============================================================================

def basis_with_n_up(N: int, n_up: int):
    """Integer encodings of all basis states with `n_up` up-spins."""
    out = []
    for positions in combinations(range(N), n_up):
        s = 0
        for p in positions:
            s |= (1 << p)
        out.append(s)
    return out


def build_h_block(N, bonds, J, n_up):
    """Build sparse Heisenberg Hamiltonian H_J = J Σ S_i·S_j inside the
    Sz_total = n_up − N/2 sector (Zeeman is added linearly afterwards)."""
    states = basis_with_n_up(N, n_up)
    state_to_idx = {s: i for i, s in enumerate(states)}
    dim = len(states)
    if dim == 0:
        return sparse.csr_matrix((0, 0))

    rows, cols, vals = [], [], []
    for idx, state in enumerate(states):
        diag = 0.0
        for i, j in bonds:
            si = 0.5 if (state >> i) & 1 else -0.5
            sj = 0.5 if (state >> j) & 1 else -0.5
            diag += J * si * sj
        rows.append(idx); cols.append(idx); vals.append(diag)

        for i, j in bonds:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                new_state = state ^ ((1 << i) | (1 << j))
                rows.append(state_to_idx[new_state])
                cols.append(idx)
                vals.append(J / 2.0)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def diagonalize_block(Hm: sparse.csr_matrix, k_max: int):
    """Return up to k_max lowest eigenvalues of Hm (sorted ascending)."""
    dim = Hm.shape[0]
    if dim == 0:
        return np.array([])
    if dim <= 250:
        ev = np.linalg.eigvalsh(Hm.toarray())
        return ev[:k_max]
    k = min(k_max, dim - 2)
    ev, _ = eigsh(Hm, k=k, which='SA')
    return np.sort(ev)


def make_lattice(kind, params, pbc=True):
    """Return (N, bonds, positions) for a given lattice."""
    if kind == "1D Chain":
        N = int(params['N'])
        bonds = [(i, i + 1) for i in range(N - 1)]
        if pbc and N > 2:
            bonds.append((N - 1, 0))
        positions = {i: (float(i), 0.0) for i in range(N)}
        return N, bonds, positions

    if kind == "Triangle (3 sites)":
        N = 3
        bonds = [(0, 1), (1, 2), (0, 2)]
        positions = {0: (0, 0), 1: (1, 0), 2: (0.5, np.sqrt(3) / 2)}
        return N, bonds, positions

    if kind == "2D Square":
        Lx = int(params['Lx']); Ly = int(params['Ly'])
        N = Lx * Ly
        bonds = []
        for x in range(Lx):
            for y in range(Ly):
                i = x + y * Lx
                # nearest-neighbor in +x direction
                if x < Lx - 1:
                    j = (x + 1) + y * Lx
                    bonds.append(tuple(sorted((i, j))))
                elif pbc and Lx > 2:
                    j = 0 + y * Lx
                    bonds.append(tuple(sorted((i, j))))
                # nearest-neighbor in +y direction
                if y < Ly - 1:
                    j = x + (y + 1) * Lx
                    bonds.append(tuple(sorted((i, j))))
                elif pbc and Ly > 2:
                    j = x + 0
                    bonds.append(tuple(sorted((i, j))))
        bonds = sorted(set(bonds))
        positions = {x + y * Lx: (x, y) for x in range(Lx) for y in range(Ly)}
        return N, bonds, positions

    raise ValueError(f"Unknown lattice {kind}")


def solve_heisenberg(N, bonds, J, k_per_sector=4, progress_cb=None):
    """Diagonalize H_J in every Sz sector at H=0 and return lowest eigenvalues."""
    sector_eigs = {}    # {n_up: eigenvalues_at_H0}
    sector_dims = {}
    for n_up in range(N + 1):
        Hm = build_h_block(N, bonds, J, n_up)
        sector_dims[n_up] = Hm.shape[0]
        sector_eigs[n_up] = diagonalize_block(Hm, k_per_sector)
        if progress_cb is not None:
            progress_cb(n_up + 1, N + 1)
    return sector_eigs, sector_dims


def assemble_spectrum(N, sector_eigs, H_array):
    """Add Zeeman shift H * Sz_total to each sector and stack into one array."""
    rows = []
    for n_up, evs in sector_eigs.items():
        sz = n_up - N / 2.0
        for k, e0 in enumerate(evs):
            energies = e0 + H_array * sz
            rows.append({'sz': sz, 'k': k, 'E0': e0, 'E_of_H': energies})
    return rows


def ground_state_curve(N, sector_eigs, H_array):
    """Return GS energy and Sz at each H value."""
    E_gs = np.full_like(H_array, np.inf, dtype=float)
    Sz_gs = np.zeros_like(H_array, dtype=float)
    for n_up, evs in sector_eigs.items():
        if len(evs) == 0:
            continue
        sz = n_up - N / 2.0
        E_low = evs[0] + H_array * sz
        better = E_low < E_gs - 1e-10
        E_gs = np.where(better, E_low, E_gs)
        Sz_gs = np.where(better, sz, Sz_gs)
    return E_gs, Sz_gs


# ============================================================================
# ============   PHASE 2 :  BOSE-FERMI MIXTURE (TF + LDA)   ==================
# ============================================================================
# Natural units: ℏ = 1, m_B = 1, ω_B = 1.
#   length    : a_ho_B = √(ℏ/m_Bω_B)
#   energy    : ℏω_B
#   density   : 1/a_ho_B³
# Inputs:
#   gB           = 4π a_B/a_ho_B                              (boson-boson)
#   gBF          = 2π a_BF/a_ho_B · m_B/μ                     (boson-fermion)
#   mass_ratio   = m_F/m_B
#   omega_ratio  = ω_F/ω_B
# Equations solved self-consistently:
#   n_B(r) = max( [μ_B − V_B(r) − gBF n_F(r)] / gB , 0 )                (TF)
#   n_F(r) = (1/6π²)·[ 2(m_F/m_B)·(μ_F − V_F(r) − gBF n_B(r)) ]^{3/2}   (LDA)
#   ∫4πr² n_B dr = N_B,  ∫4πr² n_F dr = N_F
#
# V_B(r) = ½ r²,   V_F(r) = ½ (m_F/m_B)(ω_F/ω_B)² r²

def boson_density(mu_B, n_F, V_B, gB, gBF):
    return np.maximum((mu_B - V_B - gBF * n_F) / gB, 0.0)


def fermion_density(mu_F, n_B, V_F, mass_ratio, gBF):
    arg = 2.0 * mass_ratio * (mu_F - V_F - gBF * n_B)
    arg = np.maximum(arg, 0.0)
    return (1.0 / (6.0 * np.pi ** 2)) * arg ** 1.5


_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def integrate_radial(profile, r):
    """∫ 4π r² profile dr."""
    return float(_trapezoid(4.0 * np.pi * r ** 2 * profile, r))


def find_mu(target_N, density_fn, *args):
    """Bisect on chemical potential so total particle number matches `target_N`.
    `density_fn(mu, *args)` must return a 1-D density array on the radial grid."""
    r = args[-1]                                              # last arg is r-grid

    def shortfall(mu):
        n = density_fn(mu, *args[:-1])
        return integrate_radial(n, r) - target_N

    lo, hi = -50.0, 1.0
    while shortfall(hi) < 0 and hi < 1e6:
        hi *= 2.0
    while shortfall(lo) > 0 and lo > -1e6:
        lo *= 2.0
    try:
        return brentq(shortfall, lo, hi, xtol=1e-8, rtol=1e-6)
    except ValueError:                                         # numerical edge
        return hi


def solve_bf_mixture(NB, NF, gB, gBF, mass_ratio=1.0, omega_ratio=1.0,
                     r_max=None, n_grid=1000, mixing=0.4, max_iter=400, tol=1e-7):
    """Self-consistent Thomas-Fermi + LDA solution for the trapped mixture."""

    # Estimate a generous radial extent from non-interacting TF radii
    R_B = (15.0 * NB * max(gB, 1e-6) / (4.0 * np.pi)) ** (1.0 / 5.0)
    mu_F_est = omega_ratio * (3.0 * NF) ** (1.0 / 3.0)
    R_F = np.sqrt(2.0 * mu_F_est / max(mass_ratio * omega_ratio ** 2, 1e-6))
    if r_max is None:
        r_max = 1.6 * max(R_B, R_F, 1.0)

    r = np.linspace(1e-4, r_max, n_grid)
    V_B = 0.5 * r ** 2
    V_F = 0.5 * mass_ratio * omega_ratio ** 2 * r ** 2

    # ---- initialize each species independently
    boson_args = (np.zeros_like(r), V_B, gB, gBF, r)
    mu_B = find_mu(NB, boson_density, *boson_args)
    n_B = boson_density(mu_B, np.zeros_like(r), V_B, gB, gBF)

    fermion_args = (np.zeros_like(r), V_F, mass_ratio, gBF, r)
    mu_F = find_mu(NF, fermion_density, *fermion_args)
    n_F = fermion_density(mu_F, np.zeros_like(r), V_F, mass_ratio, gBF)

    history = []
    for it in range(max_iter):
        n_B_old, n_F_old = n_B.copy(), n_F.copy()

        mu_B = find_mu(NB, boson_density, n_F, V_B, gB, gBF, r)
        n_B_new = boson_density(mu_B, n_F, V_B, gB, gBF)
        n_B = mixing * n_B_new + (1 - mixing) * n_B

        mu_F = find_mu(NF, fermion_density, n_B, V_F, mass_ratio, gBF, r)
        n_F_new = fermion_density(mu_F, n_B, V_F, mass_ratio, gBF)
        n_F = mixing * n_F_new + (1 - mixing) * n_F

        delta = (np.max(np.abs(n_B - n_B_old)) / max(np.max(n_B), 1e-15) +
                 np.max(np.abs(n_F - n_F_old)) / max(np.max(n_F), 1e-15))
        history.append(delta)
        if delta < tol:
            break

    # ---- diagnostics
    R_TF_B = r[np.where(n_B > 1e-6 * np.max(n_B))[0][-1]] if np.max(n_B) > 0 else 0.0
    R_TF_F = r[np.where(n_F > 1e-6 * np.max(n_F))[0][-1]] if np.max(n_F) > 0 else 0.0
    NB_check = integrate_radial(n_B, r)
    NF_check = integrate_radial(n_F, r)

    return {
        'r': r, 'n_B': n_B, 'n_F': n_F,
        'V_B': V_B, 'V_F': V_F,
        'mu_B': mu_B, 'mu_F': mu_F,
        'R_TF_B': R_TF_B, 'R_TF_F': R_TF_F,
        'n_B_0': float(n_B[0]), 'n_F_0': float(n_F[0]),
        'N_B': NB_check, 'N_F': NF_check,
        'iterations': it + 1, 'converged': delta < tol,
        'final_delta': float(delta),
    }


# ============================================================================
# ============   STREAMLIT  UI   =============================================
# ============================================================================

st.set_page_config(page_title="Many-Body Quantum Simulator",
                   layout="wide",
                   page_icon="⚛️")

st.title("⚛️ Many-Body Quantum Simulator")
st.caption("P452 Project 2 — Heisenberg ED  +  Bose-Fermi mixture in a trap")

mode = st.sidebar.radio(
    "Choose simulation",
    ["Phase 1: Heisenberg model", "Phase 2: Bose-Fermi mixture"],
    label_visibility="visible",
)
st.sidebar.divider()

# ----------------------------------------------------------------------------
# ----------------------  PHASE 1 UI  ----------------------------------------
# ----------------------------------------------------------------------------
if mode == "Phase 1: Heisenberg model":
    st.sidebar.subheader("Lattice")
    lattice_kind = st.sidebar.selectbox(
        "Geometry",
        ["1D Chain", "Triangle (3 sites)", "2D Square"],
    )
    lattice_params = {}
    pbc = True
    if lattice_kind == "1D Chain":
        lattice_params['N'] = st.sidebar.slider("N (sites)", 2, 18, 8)
        pbc = st.sidebar.checkbox("Periodic boundary conditions", True)
    elif lattice_kind == "2D Square":
        lattice_params['Lx'] = st.sidebar.slider("Lx", 2, 5, 2)
        lattice_params['Ly'] = st.sidebar.slider("Ly", 2, 5, 2)
        pbc = st.sidebar.checkbox("Periodic boundary conditions", True)

    N_preview, bonds_preview, _ = make_lattice(lattice_kind, lattice_params, pbc)
    st.sidebar.caption(f"N = {N_preview} sites, {len(bonds_preview)} bonds, "
                       f"Hilbert dim = {2 ** N_preview:,}")

    st.sidebar.subheader("Couplings")
    J = st.sidebar.number_input("Exchange J", 0.0, 10.0, 1.0, 0.1,
                                help="J > 0 is anti-ferromagnetic.")
    H_max = st.sidebar.slider("Max field H/J", 0.5, 10.0, 4.0, 0.5)
    n_H = st.sidebar.slider("Field-grid resolution", 21, 401, 161, 20)

    k_per_sector = st.sidebar.slider("Eigenvalues per Sz sector", 1, 8, 3,
                                     help="Lowest k from each block.")

    run_btn = st.sidebar.button("▶ Run simulation", type="primary",
                                use_container_width=True)

    # ----- main panel
    st.header("Heisenberg model  Ĥ = J Σ Sᵢ·Sⱼ + H Σ Sᵢᶻ")

    N, bonds, positions = make_lattice(lattice_kind, lattice_params, pbc)
    largest_block = max(_comb(N, n_up) for n_up in range(N + 1))

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Sites N", N)
    info_col2.metric("Bonds", len(bonds))
    info_col3.metric("Hilbert dim", f"{2 ** N:,}")
    info_col4.metric("Largest Sz block", f"{largest_block:,}")

    # ---- lattice picture
    fig_lat, ax_lat = plt.subplots(figsize=(4.5, 3.5))
    xs = [positions[i][0] for i in range(N)]
    ys = [positions[i][1] for i in range(N)]
    for (i, j) in bonds:
        ax_lat.plot([positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    color='#888', lw=1.5, zorder=1)
    ax_lat.scatter(xs, ys, s=300, color='#3A86FF', zorder=2, edgecolor='k')
    for i in range(N):
        ax_lat.text(positions[i][0], positions[i][1], str(i), ha='center',
                    va='center', color='white', fontsize=9, zorder=3)
    ax_lat.set_aspect('equal')
    ax_lat.set_title(f"{lattice_kind} (PBC = {pbc})")
    ax_lat.axis('off')
    with st.expander("Lattice geometry", expanded=False):
        st.pyplot(fig_lat)

    if run_btn:
        if 2 ** N > 5e5 and largest_block > 5e4:
            st.warning(f"Largest Sz block has {largest_block:,} states — "
                       "this may take a minute or two.")

        H_array = np.linspace(0, H_max * J, n_H)

        with st.spinner("Diagonalising every Sz sector..."):
            progress = st.progress(0.0, text="0/0 sectors")

            def cb(done, total):
                progress.progress(done / total, text=f"{done}/{total} sectors")

            sector_eigs, sector_dims = solve_heisenberg(
                N, bonds, J, k_per_sector=k_per_sector, progress_cb=cb
            )
            progress.empty()

        # ----- assemble plotting data
        spectrum_rows = assemble_spectrum(N, sector_eigs, H_array)
        E_gs, Sz_gs = ground_state_curve(N, sector_eigs, H_array)

        st.success(f"Done. Lowest energy at H=0: E₀ = {E_gs[0]:.6f}, "
                   f"Sz_total = {Sz_gs[0]}")

        # ============== PLOTS =================================================
        col_a, col_b = st.columns(2)

        # -- (a) energy spectrum vs H/J
        fig1, ax1 = plt.subplots(figsize=(6, 4.4))
        cmap = plt.cm.viridis
        sz_vals = sorted({row['sz'] for row in spectrum_rows})
        sz_norm = matplotlib.colors.Normalize(vmin=min(sz_vals), vmax=max(sz_vals))
        plotted_sz = set()
        for row in spectrum_rows:
            color = cmap(sz_norm(row['sz']))
            label = f"Sz={row['sz']:+g}" if row['sz'] not in plotted_sz else None
            plotted_sz.add(row['sz'])
            ax1.plot(H_array / J, row['E_of_H'], color=color, lw=1.0,
                     alpha=0.7, label=label)
        ax1.plot(H_array / J, E_gs, color='red', lw=2.0,
                 label='ground state', zorder=10)
        ax1.set_xlabel("H / J")
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy spectrum (lowest eigenvalues per Sz sector)")
        ax1.legend(fontsize=7, loc='upper left',
                   bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig1.tight_layout()
        col_a.pyplot(fig1)

        # -- (b) magnetization vs H/J (per site)
        Mz_per_site = Sz_gs / N
        fig2, ax2 = plt.subplots(figsize=(6, 4.4))
        ax2.plot(H_array / J, Mz_per_site, color='#FB5607', lw=2.0)
        ax2.set_xlabel("H / J")
        ax2.set_ylabel(r"$\langle M^z \rangle / N$")
        ax2.set_title("Ground-state magnetisation per site")
        ax2.set_ylim(min(Mz_per_site) - 0.05, max(Mz_per_site) + 0.05)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        col_b.pyplot(fig2)

        # ============== DATA TABLE ============================================
        st.subheader("Sector summary (at H = 0)")
        rows = []
        for n_up in range(N + 1):
            sz = n_up - N / 2.0
            evs = sector_eigs[n_up]
            row = {
                'n_up':         n_up,
                'Sz_total':     f"{sz:+g}",
                'block_dim':    sector_dims[n_up],
                'E_lowest':     float(evs[0]) if len(evs) else np.nan,
            }
            for k in range(1, k_per_sector):
                row[f'E_{k}'] = float(evs[k]) if k < len(evs) else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({c: "{:.6f}" for c in df.columns
                                      if c.startswith('E_') or c == 'E_lowest'}),
                     use_container_width=True, hide_index=True)

        st.subheader("Ground-state plateau structure")
        plateau_rows = []
        prev_sz = None
        for h, sz, e in zip(H_array, Sz_gs, E_gs):
            if sz != prev_sz:
                plateau_rows.append({
                    'H/J at plateau start': h / J,
                    'Sz_total': f"{sz:+g}",
                    'Mz / N':   sz / N,
                    'E_gs at start': e,
                })
                prev_sz = sz
        df_plateau = pd.DataFrame(plateau_rows)
        st.dataframe(
            df_plateau.style.format({
                'H/J at plateau start': "{:.4f}",
                'Mz / N':              "{:.4f}",
                'E_gs at start':       "{:.6f}",
            }),
            use_container_width=True, hide_index=True
        )

        # ----- offer CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download sector data (CSV)", csv,
                           file_name=f"heisenberg_{lattice_kind}_N{N}.csv",
                           mime="text/csv")


# ----------------------------------------------------------------------------
# ----------------------  PHASE 2 UI  ----------------------------------------
# ----------------------------------------------------------------------------
else:
    st.sidebar.subheader("Particle numbers")
    NB = st.sidebar.number_input("N_B  (bosons)", 100, 10_000_000, 100_000, 1000)
    NF = st.sidebar.number_input("N_F  (fermions)", 100, 10_000_000, 10_000, 1000)

    st.sidebar.subheader("Mass / trap ratios")
    mass_ratio = st.sidebar.number_input("m_F / m_B", 0.05, 20.0, 40.0 / 87.0, 0.01,
                                         format="%.4f",
                                         help="Default 40/87 = ⁴⁰K / ⁸⁷Rb.")
    omega_ratio = st.sidebar.number_input("ω_F / ω_B", 0.1, 10.0, 1.0, 0.1)

    st.sidebar.subheader("Couplings (HO units)")
    gB = st.sidebar.number_input("g_B  = 4π a_B / a_ho",
                                 0.0001, 5.0, 0.05, 0.01, format="%.4f",
                                 help="Default ≈ 0.05 for a_B≈100 a₀, ω/2π≈100 Hz.")
    gBF = st.sidebar.slider("g_BF (interspecies)",
                            -1.0, 1.0, 0.0, 0.005, format="%.3f",
                            help=r"g_BF = 2π a_BF/a_ho · m_B/μ. "
                                 "Negative → attractive, positive → repulsive.")

    st.sidebar.subheader("Numerical")
    n_grid = st.sidebar.slider("Radial grid points", 200, 4000, 1000, 100)
    auto_compare = st.sidebar.checkbox("Compare three g_BF regimes",
                                       value=True,
                                       help="Side-by-side: attractive, none, repulsive.")
    run_btn = st.sidebar.button("▶ Run simulation", type="primary",
                                use_container_width=True)

    st.header("Bose-Fermi mixture in a spherical trap (TF + LDA)")
    st.caption(r"Solving self-consistently:  "
               r"$n_B = \max((\mu_B - V_B - g_{BF}n_F)/g_B,0)$,  "
               r"$E_F[n_F] = \mu_F - V_F - g_{BF}n_B$")

    # -- diagnostics row (before simulation runs)
    det_M = gB * (2.0 / 3.0) * (6.0 * np.pi ** 2) ** (2.0 / 3.0) * 0  # placeholder
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    diag_col1.metric("g_B", f"{gB:.4f}")
    diag_col2.metric("g_BF", f"{gBF:+.4f}")
    diag_col3.metric("m_F / m_B", f"{mass_ratio:.4f}")

    if run_btn:
        runs = []
        if auto_compare:
            for label, g in [("Attractive (g_BF = -|g|)", -abs(gBF) if gBF != 0 else -0.05),
                             ("Non-interacting (g_BF = 0)", 0.0),
                             ("Repulsive (g_BF = +|g|)", abs(gBF) if gBF != 0 else +0.05)]:
                with st.spinner(f"Solving {label}..."):
                    runs.append((label, solve_bf_mixture(
                        NB, NF, gB, g, mass_ratio, omega_ratio, n_grid=n_grid)))
        else:
            with st.spinner("Solving self-consistent equations..."):
                runs.append((f"g_BF = {gBF:+.4f}", solve_bf_mixture(
                    NB, NF, gB, gBF, mass_ratio, omega_ratio, n_grid=n_grid)))

        # -- summary table
        rows = []
        for label, sol in runs:
            rows.append({
                'case':          label,
                'g_BF':          gBF if not auto_compare else (
                                 -abs(gBF) if 'Attractive' in label else
                                 0.0       if 'Non' in label else
                                 +abs(gBF)),
                'iterations':    sol['iterations'],
                'converged':     "✓" if sol['converged'] else "✗",
                'μ_B':           sol['mu_B'],
                'μ_F':           sol['mu_F'],
                'R_TF_B':        sol['R_TF_B'],
                'R_TF_F':        sol['R_TF_F'],
                'n_B(0)':        sol['n_B_0'],
                'n_F(0)':        sol['n_F_0'],
                'N_B (check)':   sol['N_B'],
                'N_F (check)':   sol['N_F'],
            })
        df = pd.DataFrame(rows)
        # nicer formatting
        fmt = {
            'g_BF':         "{:+.4f}",
            'μ_B':          "{:.3f}", 'μ_F': "{:.3f}",
            'R_TF_B':       "{:.3f}", 'R_TF_F': "{:.3f}",
            'n_B(0)':       "{:.4f}", 'n_F(0)': "{:.4f}",
            'N_B (check)':  "{:.1f}", 'N_F (check)': "{:.1f}",
        }

        # ---- plot 1 : density profiles ---------------------------------------
        if len(runs) == 1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            label, sol = runs[0]
            ax.plot(sol['r'], sol['n_B'], color='#3A86FF', lw=2.0, label=r"$n_B(r)$")
            ax.plot(sol['r'], sol['n_F'], color='#FB5607', lw=2.0, label=r"$n_F(r)$")
            ax.set_xlabel(r"$r / a_{\mathrm{ho},B}$")
            ax.set_ylabel("Density  (1 / $a_{\\mathrm{ho},B}^3$)")
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(sol['R_TF_B'], sol['R_TF_F']) * 1.15 + 0.1)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
            r_max_plot = max(max(sol['R_TF_B'], sol['R_TF_F']) for _, sol in runs) * 1.15
            for (label, sol), ax in zip(runs, axes):
                ax.plot(sol['r'], sol['n_B'], color='#3A86FF', lw=2.0,
                        label=r"$n_B(r)$")
                ax.plot(sol['r'], sol['n_F'], color='#FB5607', lw=2.0,
                        label=r"$n_F(r)$")
                ax.set_xlabel(r"$r / a_{\mathrm{ho},B}$")
                ax.set_title(label, fontsize=10)
                ax.set_xlim(0, r_max_plot)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
            axes[0].set_ylabel(r"Density  (1 / $a_{\mathrm{ho},B}^3$)")
            fig.tight_layout()
            st.pyplot(fig)

        # ---- plot 2 : overlay all profiles + central density vs g_BF ---------
        fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.4))
        colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, max(len(runs), 3)))

        for (label, sol), c in zip(runs, colors):
            ax_a.plot(sol['r'], sol['n_B'], color=c, lw=1.8,
                      label=label.split('(')[0].strip(), linestyle='-')
            ax_a.plot(sol['r'], sol['n_F'], color=c, lw=1.8, linestyle='--')
        ax_a.set_xlabel(r"$r / a_{\mathrm{ho},B}$")
        ax_a.set_ylabel("Density")
        ax_a.set_title(r"Density profiles  (solid: $n_B$, dashed: $n_F$)")
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

        # central density and TF radii vs g_BF (only meaningful in compare mode)
        if auto_compare:
            g_list = [r['g_BF'] for r in rows]
            ax_b.plot(g_list, [r['n_B(0)'] for r in rows], 'o-',
                      color='#3A86FF', label=r"$n_B(0)$")
            ax_b.plot(g_list, [r['n_F(0)'] for r in rows], 's-',
                      color='#FB5607', label=r"$n_F(0)$")
            ax_b.set_xlabel(r"$g_{BF}$")
            ax_b.set_ylabel("Central density")
            ax_b.set_title(r"Central density vs $g_{BF}$")
            ax_b.legend()
            ax_b.grid(True, alpha=0.3)
        else:
            sol = runs[0][1]
            ax_b.plot(sol['r'], sol['V_B'], color='#3A86FF', lw=1.5,
                      label=r"$V_B(r)$")
            ax_b.plot(sol['r'], sol['V_F'], color='#FB5607', lw=1.5,
                      label=r"$V_F(r)$")
            ax_b.axhline(sol['mu_B'], color='#3A86FF', lw=1.0, linestyle=':',
                         label=fr"$\mu_B$={sol['mu_B']:.2f}")
            ax_b.axhline(sol['mu_F'], color='#FB5607', lw=1.0, linestyle=':',
                         label=fr"$\mu_F$={sol['mu_F']:.2f}")
            ax_b.set_xlabel(r"$r / a_{\mathrm{ho},B}$")
            ax_b.set_ylabel("Energy")
            ax_b.set_title("Trap potentials and chemical potentials")
            ax_b.legend(fontsize=8)
            ax_b.grid(True, alpha=0.3)

        fig2.tight_layout()
        st.pyplot(fig2)

        # ---- summary data table ---------------------------------------------
        st.subheader("Numerical summary")
        st.dataframe(df.style.format(fmt), use_container_width=True,
                     hide_index=True)

        # CSV download with raw profiles for the first run
        first_label, first_sol = runs[0]
        prof_df = pd.DataFrame({
            'r':   first_sol['r'],
            'n_B': first_sol['n_B'],
            'n_F': first_sol['n_F'],
            'V_B': first_sol['V_B'],
            'V_F': first_sol['V_F'],
        })
        st.download_button("⬇️ Download density profile (CSV)",
                           prof_df.to_csv(index=False).encode(),
                           file_name="bf_mixture_profile.csv",
                           mime="text/csv")

        # ---- stability remark -----------------------------------------------
        # Stability criterion : g_B · ∂E_F/∂n_F  > g_BF²
        # ∂E_F/∂n_F at fermion central density
        st.subheader("Stability / phase-separation check")
        rows_stab = []
        for label, sol in runs:
            n_F0 = sol['n_F_0']
            if n_F0 > 0:
                dEF_dn = (1.0 / mass_ratio) * (np.pi ** 2) ** (2.0 / 3.0) \
                         * (6.0 * n_F0) ** (2.0 / 3.0) / 3.0   # ∂E_F/∂n_F at center
            else:
                dEF_dn = np.inf
            g_eff_used = (gBF if not auto_compare else (
                          -abs(gBF) if 'Attractive' in label else
                          0.0       if 'Non' in label else
                          +abs(gBF)))
            lhs = gB * dEF_dn
            rhs = g_eff_used ** 2
            stable = lhs > rhs
            rows_stab.append({
                'case':            label,
                'g_BF':            g_eff_used,
                "g_B · ∂E_F/∂n_F": lhs,
                "g_BF²":           rhs,
                "stable?":         "✓" if stable else "✗ (collapse / phase separation)",
            })
        st.dataframe(pd.DataFrame(rows_stab).style.format({
            'g_BF':            "{:+.4f}",
            'g_B · ∂E_F/∂n_F': "{:.4f}",
            'g_BF²':           "{:.4f}",
        }), use_container_width=True, hide_index=True)

        st.caption("Stability requires  g_B·∂E_F/∂n_F > g_BF² (positive determinant of "
                   "the 2×2 stability matrix). Violation → phase separation (g_BF>0) "
                   "or mean-field collapse (g_BF<0).")


# ----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown("""
**P452 Project 2** — many-body simulator.

Phase 1 uses ED in Sz blocks (sparse Lanczos for N≳14).
Phase 2 uses TF (bosons) + LDA (fermions) in a spherical trap.

Working units in Phase 2: ℏ = m_B = ω_B = 1.
""")
