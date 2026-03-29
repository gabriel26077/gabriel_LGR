"""
LGR (Lugar das Raízes / Root Locus) Calculator - Streamlit App
A didactic step-by-step Root Locus calculator.
"""

import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Calculadora LGR - Lugar das Raízes",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Calculadora do Lugar das Raízes (Root Locus)")
st.markdown("""
Ferramenta didática para cálculo e visualização do **Lugar Geométrico das Raízes (LGR)**,
mostrando todos os passos do algoritmo dos 12 passos.
""")

# ============================================================
# Helper functions
# ============================================================

def poly_to_latex(coeffs, var='s'):
    """Convert numpy polynomial coefficients to LaTeX string."""
    if not coeffs.any():
        return "0"
    latex_parts = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        term = ""
        if coeff > 0 and latex_parts:
            term += "+"
        elif coeff < 0:
            term += "-"
        coeff = abs(coeff)
        if coeff != 1 or (degree - i) == 0:
            term += str(int(coeff))
        if (degree - i) > 0:
            term += var
            if (degree - i) > 1:
                term += f"^{{{degree - i}}}"
        latex_parts.append(term)
    return "".join(latex_parts).replace("+-", "-").replace("++", "+")


def array_to_sym(coeffs, var):
    """Convert numpy coefficient array to sympy polynomial."""
    return sum(c * var**i for i, c in enumerate(reversed(coeffs)))


def fatorar_numerico(polinomio, var):
    """Numerically factorize a polynomial."""
    if sp.degree(polinomio, var) < 1:
        return polinomio
    roots = sp.nroots(polinomio)
    terms = [var - sp.N(r, 3, chop=True) for r in roots]
    leading_coeff = sp.LC(polinomio, var)
    return leading_coeff * sp.Mul(*terms, evaluate=False)


def compute_real_axis_segments(all_poles, all_zeros):
    """Compute which segments of the real axis belong to the root locus."""
    poles_real = [p.real for p in all_poles if abs(p.imag) < 1e-7]
    zeros_real = [z.real for z in all_zeros if abs(z.imag) < 1e-7]
    all_real_points = sorted(list(set(zeros_real + poles_real)))

    segments = []
    if all_real_points:
        for i in range(len(all_real_points)):
            if i < len(all_real_points) - 1:
                test_point = (all_real_points[i] + all_real_points[i+1]) / 2
            else:
                test_point = all_real_points[i] + 0.5

            count_right = (sum(1 for p in poles_real if p > test_point) +
                           sum(1 for z in zeros_real if z > test_point))

            if count_right % 2 != 0:
                start = all_real_points[i]
                end = all_real_points[i+1] if i < len(all_real_points) - 1 else all_real_points[i] - 10
                segments.append((start, end))
    return segments


def plot_base_lgr(ax, all_poles, all_zeros, rl_segments, all_roots_data=None):
    """Draw the base LGR plot elements reused in multiple steps."""
    # Draw numerical root locus in background if available
    if all_roots_data is not None:
        for i in range(all_roots_data.shape[1]):
            ax.plot(np.real(all_roots_data[:, i]), np.imag(all_roots_data[:, i]),
                    linewidth=1.5, alpha=0.3, color='gray')

    # Poles and zeros
    ax.plot(np.real(all_poles), np.imag(all_poles), 'x', markersize=12, color='red',
            markeredgewidth=3, label='Polos')
    if all_zeros:
        ax.plot(np.real(all_zeros), np.imag(all_zeros), 'o', markersize=10, color='green',
                fillstyle='none', markeredgewidth=2, label='Zeros')

    # Real axis segments
    for i, (start, end) in enumerate(rl_segments):
        ax.plot([start, end], [0, 0], color='blue', linewidth=4,
                solid_capstyle='round', alpha=0.6, label='LGR Eixo Real' if i == 0 else "")

    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)


def setup_lgr_axes(ax, all_poles, all_zeros, rl_segments, extra_points=None, title=''):
    """Set up axes limits and style for an LGR plot."""
    all_x = [p.real for p in all_poles] + [z.real for z in all_zeros]
    for s_seg, e_seg in rl_segments:
        all_x.extend([s_seg, e_seg])
    if extra_points:
        all_x.extend(extra_points)
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        x_span = x_max - x_min
        pad = x_span * 0.2 if x_span > 0 else 3.0
        ax.set_xlim(x_min - pad, max(x_max + pad, 2))
        y_coords = [abs(p.imag) for p in all_poles] + [abs(z.imag) for z in all_zeros]
        y_limit = max(y_coords) + 6 if y_coords else 6
        ax.set_ylim(-y_limit, y_limit)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=12)
    ax.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')


def sort_roots_by_proximity(prev_roots, curr_roots):
    """Sort curr_roots to best match prev_roots order (nearest-neighbor)."""
    n = len(curr_roots)
    sorted_roots = np.empty_like(curr_roots)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        # Find the closest unused root in curr_roots to prev_roots[i]
        dists = np.abs(curr_roots - prev_roots[i])
        dists[used] = np.inf
        j = np.argmin(dists)
        sorted_roots[i] = curr_roots[j]
        used[j] = True
    return sorted_roots


def compute_numerical_root_locus(num, den, k_max=10000, k_points=10000):
    """Compute numerical root locus with logarithmic K spacing and root tracking."""
    # Logarithmic spacing gives much better resolution near K=0
    # where roots move quickly away from the open-loop poles
    K_vals = np.concatenate([
        [0],
        np.logspace(-3, np.log10(k_max), k_points - 1)
    ])

    # First set of roots (K=0 → open-loop poles)
    eq0 = np.polyadd(den, K_vals[0] * num)
    prev = np.roots(eq0)
    all_roots = [prev]

    for K in K_vals[1:]:
        eq = np.polyadd(den, K * num)
        curr = np.roots(eq)
        # Sort current roots to match previous step by proximity
        curr_sorted = sort_roots_by_proximity(prev, curr)
        all_roots.append(curr_sorted)
        prev = curr_sorted

    return K_vals, np.array(all_roots)


# ============================================================
# Sidebar - Input
# ============================================================
st.sidebar.header("⚙️ Parâmetros do Sistema")

st.sidebar.subheader("G(s) - Numerador")
g_num_str = st.sidebar.text_input("Coeficientes de G(s) numerador (separados por vírgula):", "1")

st.sidebar.subheader("G(s) - Denominador")
g_den_str = st.sidebar.text_input("Coeficientes de G(s) denominador:", "1, 8, 32, 0")

st.sidebar.subheader("H(s) - Numerador")
h_num_str = st.sidebar.text_input("Coeficientes de H(s) numerador:", "1")

st.sidebar.subheader("H(s) - Denominador")
h_den_str = st.sidebar.text_input("Coeficientes de H(s) denominador:", "1, 4")

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros do LGR Numérico")
k_max = st.sidebar.number_input("K máximo:", min_value=1.0, value=10000.0, step=100.0)
k_points = st.sidebar.number_input("Número de pontos:", min_value=100, value=10000, step=100)

st.sidebar.subheader("Limites do Gráfico")
y_min_input = st.sidebar.number_input("y mínimo:", value=-10.0, step=1.0)
y_max_input = st.sidebar.number_input("y máximo:", value=10.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Critério de Ângulo (Passo 11)")
s_test_real = st.sidebar.number_input("Parte real do ponto de teste:", value=-1.0)
s_test_imag = st.sidebar.number_input("Parte imaginária do ponto de teste:", value=1.0)
threshold = st.sidebar.number_input("Tolerância (graus):", min_value=0.1, value=10.0, step=1.0)


# Parse inputs
try:
    g_num = np.array([float(x.strip()) for x in g_num_str.split(",")])
    g_den = np.array([float(x.strip()) for x in g_den_str.split(",")])
    h_num = np.array([float(x.strip()) for x in h_num_str.split(",")])
    h_den = np.array([float(x.strip()) for x in h_den_str.split(",")])
except ValueError:
    st.error("❌ Erro ao interpretar os coeficientes. Use números separados por vírgula.")
    st.stop()

# Symbolic variable
s = sp.symbols('s')

# ============================================================
# Computations
# ============================================================

# Convert to symbolic
G_num_s = array_to_sym(g_num, s)
G_den_s = array_to_sym(g_den, s)
H_num_s = array_to_sym(h_num, s)
H_den_s = array_to_sym(h_den, s)

# Compute G(s)H(s) symbolically
GH_expr = (G_num_s * H_num_s) / (G_den_s * H_den_s)
P_num_sym, P_den_sym = GH_expr.as_numer_denom()
P_den_expanded_sym = sp.expand(P_den_sym)
GH_expr_expanded_den = P_num_sym / P_den_expanded_sym

# Compute combined num/den for numerical root locus
num_combined = np.polymul(g_num, h_num)
den_combined = np.polymul(g_den, h_den)

# Factored form
GH_simplified = sp.simplify(GH_expr)
P_num_simp, P_den_simp = GH_simplified.as_numer_denom()

P_num_factored = fatorar_numerico(P_num_simp, s)
P_den_factored = fatorar_numerico(P_den_simp, s)
GH_final_display = P_num_factored / P_den_factored

# Zeros and Poles
if P_num_simp.is_number:
    all_zeros = []
else:
    all_zeros = [complex(z) for z in sp.nroots(P_num_simp)]

if P_den_simp.is_number:
    all_poles = []
else:
    all_poles = [complex(p) for p in sp.nroots(P_den_simp)]

Np = len(all_poles)
Nz = len(all_zeros)
Na = Np - Nz
Ls = max(Np, Nz)

# Real axis segments
rl_segments = compute_real_axis_segments(all_poles, all_zeros)

# ============================================================
# Step Display
# ============================================================

# --- Mostrar G(s) e H(s) ---
st.header("📊 Funções de Transferência")
col_gs, col_hs = st.columns(2)
with col_gs:
    G_display = sp.Rational(1) * G_num_s / G_den_s
    st.latex(rf"G(s) = {sp.latex(G_display)}")
with col_hs:
    H_display = sp.Rational(1) * H_num_s / H_den_s
    st.latex(rf"H(s) = {sp.latex(H_display)}")

st.markdown("---")

# --- LGR Numérico ---
st.header("🔢 LGR Numérico")
st.markdown("Determinação do LGR numericamente para comparação.")

K_vals, all_roots = compute_numerical_root_locus(num_combined, den_combined, k_max, int(k_points))

fig_num, ax_num = plt.subplots(figsize=(10, 7))
for i in range(all_roots.shape[1]):
    ax_num.plot(np.real(all_roots[:, i]), np.imag(all_roots[:, i]), linewidth=2)

polos_ma = np.roots(den_combined)
ax_num.plot(np.real(polos_ma), np.imag(polos_ma), 'x', markersize=10, color='red',
            markeredgewidth=2, label='Polos (Malha Aberta)')

zeros_ma = np.roots(num_combined)
if len(zeros_ma) > 0:
    ax_num.plot(np.real(zeros_ma), np.imag(zeros_ma), 'o', markersize=8, color='green',
                fillstyle='none', markeredgewidth=2, label='Zeros (Malha Aberta)')

ax_num.axhline(0, color='black', linewidth=1)
ax_num.axvline(0, color='black', linewidth=1)
ax_num.set_title('Lugar das Raízes (Root Locus)')
ax_num.set_xlabel('Parte Real')
ax_num.set_ylabel('Parte Imaginária')
ax_num.set_ylim(y_min_input, y_max_input)
ax_num.grid(True, linestyle='--', alpha=0.7)
ax_num.legend()
st.pyplot(fig_num)

# ============================================================
# Algoritmo dos 12 Passos
# ============================================================
st.header("📝 Algoritmo dos 12 Passos")

# --- Passo 1 ---
with st.expander("**Passo 1:** Polinômio Característico", expanded=True):
    st.markdown("Escrever o polinômio característico de modo que K apareça claramente:")
    latex_final = rf"1 + G(s)H(s) = 1 + k{sp.latex(GH_expr_expanded_den)} = 1 + kP(s)"
    st.latex(latex_final)

# --- Passo 2 ---
with st.expander("**Passo 2:** Fatorar P(s) em polos e zeros"):
    st.latex(rf"P(s) = {sp.latex(GH_final_display)}")

# --- Passo 3 ---
with st.expander("**Passo 3:** Polos e Zeros"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Polos:**")
        for i, p in enumerate(all_poles):
            st.markdown(f"- $p_{{{i+1}}} = {p.real:.4f} {p.imag:+.4f}j$")
    with col2:
        st.markdown("**Zeros:**")
        if all_zeros:
            for i, z in enumerate(all_zeros):
                st.markdown(f"- $z_{{{i+1}}} = {z.real:.4f} {z.imag:+.4f}j$")
        else:
            st.markdown("*Não há zeros finitos.*")

# --- Passo 4 ---
with st.expander("**Passo 4:** Segmentos do eixo real que pertencem ao LGR"):
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ax4.plot(np.real(all_poles), np.imag(all_poles), 'x', markersize=12, color='red',
             markeredgewidth=3, label='Polos')
    if all_zeros:
        ax4.plot(np.real(all_zeros), np.imag(all_zeros), 'o', markersize=10, color='green',
                 fillstyle='none', markeredgewidth=2, label='Zeros')

    for i, (start, end) in enumerate(rl_segments):
        ax4.plot([start, end], [0, 0], color='blue', linewidth=4,
                 solid_capstyle='round', label='LGR Eixo Real' if i == 0 else "")

    ax4.axhline(0, color='black', linewidth=1.2)
    ax4.axvline(0, color='black', linewidth=1.2)

    all_x = [p.real for p in all_poles] + [z.real for z in all_zeros]
    for s_seg, e_seg in rl_segments:
        all_x.extend([s_seg, e_seg])
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        x_span = x_max - x_min
        pad = x_span * 0.1 if x_span > 0 else 1.0
        ax4.set_xlim(x_min - pad, x_max + pad)
        y_coords = [abs(p.imag) for p in all_poles] + [abs(z.imag) for z in all_zeros]
        y_limit = max(y_coords) + 1 if y_coords else 2
        ax4.set_ylim(-y_limit, y_limit)

    ax4.set_aspect('auto')
    ax4.set_title('LGR - Segmentos do Eixo Real', fontsize=14)
    ax4.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=12)
    ax4.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("**Regra:** A esquerda de números ímpares de polos ou zeros reais.")
    if rl_segments:
        for seg in rl_segments:
            st.markdown(f"- Segmento: [{seg[0]:.4f}, {seg[1]:.4f}]")
    else:
        st.markdown("*Nenhum segmento do eixo real pertence ao LGR.*")

# --- Passo 5 ---
with st.expander("**Passo 5:** Número de lugares separados"):
    st.markdown(rf"O número de polos $N_p$ é: **{Np}**")
    st.markdown(rf"O número de zeros $N_z$ é: **{Nz}**")
    st.markdown(rf"O número de lugares separados $L_s = \max\{{N_p, N_z\}}$ é: **{Ls}**")

# --- Passo 6 ---
with st.expander("**Passo 6:** Simetria"):
    st.markdown("O LGR é **simétrico em relação ao eixo real**.")
    st.markdown("Isso se deve ao fato de que raízes complexas sempre ocorrem em pares conjugados.")

# --- Passo 7 ---
with st.expander("**Passo 7:** Assíntotas e ângulos"):
    if Na > 0:
        sum_poles = sum(np.real(all_poles))
        sum_zeros = sum(np.real(all_zeros)) if Nz > 0 else 0
        sigma_A = (sum_poles - sum_zeros) / Na

        angles_deg = [((2 * q + 1) * 180) / Na for q in range(Na)]
        angles_rad = np.deg2rad(angles_deg)

        st.markdown("### Centro das assíntotas e ângulos")
        st.latex(rf"\sigma_A = \frac{{\sum (-p_j) - \sum (-z_i)}}{{n_p - n_z}}")
        st.latex(rf"\sigma_A = \frac{{({sum_poles:.2f}) - ({sum_zeros:.2f})}}{{{Na}}} = {sigma_A:.2f}")

        st.latex(rf"\phi_A = \frac{{(2q + 1)}}{{n_p - n_z}} \cdot 180^\circ")
        for q, ang in enumerate(angles_deg):
            st.markdown(rf"Para $q = {q}$: $\phi_A = {ang:.1f}°$")

        # Plot com assintotas
        fig7, ax7 = plt.subplots(figsize=(15, 6))
        ax7.plot(np.real(all_poles), np.imag(all_poles), 'x', markersize=12, color='red',
                 markeredgewidth=3, label='Polos')
        if all_zeros:
            ax7.plot(np.real(all_zeros), np.imag(all_zeros), 'o', markersize=10, color='green',
                     fillstyle='none', markeredgewidth=2, label='Zeros')

        for i, (start, end) in enumerate(rl_segments):
            ax7.plot([start, end], [0, 0], color='blue', linewidth=4,
                     solid_capstyle='round', label='LGR Eixo Real' if i == 0 else "")

        line_length = 30
        for i, angle in enumerate(angles_rad):
            dx = line_length * np.cos(angle)
            dy = line_length * np.sin(angle)
            ax7.plot([sigma_A, sigma_A + dx], [0, dy], '--', color='darkorange',
                     alpha=0.8, linewidth=2, label='Assíntotas' if i == 0 else "")

        ax7.plot(sigma_A, 0, '*', markersize=12, color='darkviolet',
                 label=r'Centroide ($\sigma_A$)')

        ax7.axhline(0, color='black', linewidth=1.2)
        ax7.axvline(0, color='black', linewidth=1.2)

        all_x = [p.real for p in all_poles] + [z.real for z in all_zeros] + [sigma_A]
        for s_seg, e_seg in rl_segments:
            all_x.extend([s_seg, e_seg])
        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            x_span = x_max - x_min
            pad = x_span * 0.2 if x_span > 0 else 3.0
            ax7.set_xlim(x_min - pad, max(x_max + pad, 2))
            y_coords = [abs(p.imag) for p in all_poles] + [abs(z.imag) for z in all_zeros]
            y_limit = max(y_coords) + 6 if y_coords else 6
            ax7.set_ylim(-y_limit, y_limit)

        ax7.set_aspect('auto')
        ax7.set_title('LGR com Assíntotas', fontsize=14)
        ax7.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=12)
        ax7.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=12)
        ax7.grid(True, linestyle=':', alpha=0.6)
        ax7.legend(loc='upper right')
        plt.tight_layout()
        st.pyplot(fig7)
    else:
        st.markdown("**Não há assíntotas para o infinito.** $n_p \\le n_z$")

# --- Passo 8: Pontos de Saída/Entrada (Descolamento) ---
with st.expander("**Passo 8:** Pontos de Saída/Entrada no Eixo Real"):
    st.markdown("### Pontos de descolamento (Breakaway/Break-in)")

    # K = -D(s)/N(s), dK/ds = 0
    N_s = P_num_sym
    D_s = P_den_sym
    K_expr_break = -D_s / N_s

    dN_ds = sp.diff(N_s, s)
    dD_ds = sp.diff(D_s, s)
    break_eq = dD_ds * N_s - D_s * dN_ds

    st.markdown(r"**1º Fazer $K = p(s)$:**")
    st.markdown(r"A partir da equação característica, isolamos $K$:")
    st.latex(rf"K = p(s) = -\frac{{D(s)}}{{N(s)}} = {sp.latex(K_expr_break)}")

    st.markdown(r"**2º Determinar as raízes de $\frac{dp(s)}{ds} = 0$:**")
    dK_ds_simplified = sp.cancel(sp.diff(K_expr_break, s))
    st.latex(rf"\frac{{dp(s)}}{{ds}} = {sp.latex(dK_ds_simplified)}")

    st.markdown(r"Para a derivada ser zero, basta que o polinômio do numerador seja zero:")
    st.latex(rf"{sp.latex(sp.expand(break_eq))} = 0")

    # Compute break roots
    try:
        break_roots_sympy = sp.nroots(break_eq)
        break_roots_complex = [complex(r) for r in break_roots_sympy]
    except Exception:
        break_roots_complex = []

    tol = 1e-5
    valid_break_points = []
    for r in break_roots_complex:
        if abs(r.imag) < tol:
            real_val = r.real
            is_on_lgr_bp = False
            for start, end in rl_segments:
                seg_min, seg_max = min(start, end), max(start, end)
                if seg_min - tol <= real_val <= seg_max + tol:
                    is_on_lgr_bp = True
                    break
            if is_on_lgr_bp:
                valid_break_points.append(real_val)

    valid_break_points = sorted(list(set([round(p, 4) for p in valid_break_points])))

    all_roots_str = ", ".join([f"{r.real:.4f} + {r.imag:.4f}j" if abs(r.imag) > tol else f"{r.real:.4f}" for r in break_roots_complex])
    st.markdown(rf"**Todas as raízes calculadas:** $s = [{all_roots_str}]$")

    if valid_break_points:
        points_str = ", ".join([str(p) for p in valid_break_points])
        st.success(rf"**Raízes válidas (pertencem ao LGR no eixo real):** $s = {points_str}$")
    else:
        st.info("**Não há raízes válidas no eixo real para este sistema.**")

    # Plot: breakaway points
    fig8, ax8 = plt.subplots(figsize=(15, 6))
    ax8.plot(np.real(all_poles), np.imag(all_poles), 'x', markersize=12, color='red',
             markeredgewidth=3, label='Polos (P)')
    if all_zeros:
        ax8.plot(np.real(all_zeros), np.imag(all_zeros), 'o', markersize=10, color='green',
                 fillstyle='none', markeredgewidth=2, label='Zeros (Z)')
    for i, (start, end) in enumerate(rl_segments):
        ax8.plot([start, end], [0, 0], color='blue', linewidth=4,
                 solid_capstyle='round', label='LGR Eixo Real' if i == 0 else "")
    if Na > 0:
        line_length = 30
        for i, angle in enumerate(angles_rad):
            dx = line_length * np.cos(angle)
            dy = line_length * np.sin(angle)
            ax8.plot([sigma_A, sigma_A + dx], [0, dy], '--', color='darkorange',
                     alpha=0.5, linewidth=2, label='Assíntotas' if i == 0 else "")
    if valid_break_points:
        ax8.plot(valid_break_points, [0]*len(valid_break_points), 'd', markersize=10,
                 color='magenta', markeredgewidth=2, label='Pontos de Saída/Entrada')
        for p in valid_break_points:
            ax8.text(p, 0.5, f'{p:.2f}', color='magenta', fontsize=11, fontweight='bold',
                     ha='center', bbox=dict(facecolor='white', edgecolor='magenta',
                     boxstyle='round,pad=0.2', alpha=0.8))

    ax8.axhline(0, color='black', linewidth=1.2)
    ax8.axvline(0, color='black', linewidth=1.2)
    all_x8 = [p.real for p in all_poles] + [z.real for z in all_zeros]
    if Na > 0:
        all_x8.append(sigma_A)
    all_x8.extend(valid_break_points)
    for start, end in rl_segments:
        all_x8.extend([start, end])
    if all_x8:
        x_min8, x_max8 = min(all_x8), max(all_x8)
        x_span8 = x_max8 - x_min8
        pad8 = x_span8 * 0.2 if x_span8 > 0 else 3.0
        ax8.set_xlim(x_min8 - pad8, max(x_max8 + pad8, 2))
        y_coords8 = [abs(p.imag) for p in all_poles] + [abs(z.imag) for z in all_zeros]
        y_limit8 = max(y_coords8) + 6 if y_coords8 else 6
        ax8.set_ylim(-y_limit8, y_limit8)
    ax8.set_aspect('auto')
    ax8.set_title('Lugar das Raízes (com Pontos de Descolamento)', fontsize=14)
    ax8.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=12)
    ax8.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=12)
    ax8.grid(True, linestyle=':', alpha=0.6)
    ax8.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig8)

# --- Passo 9: Cruzamento com o Eixo Imaginário (Routh-Hurwitz) ---
with st.expander("**Passo 9:** Cruzamento com o Eixo Imaginário (Routh-Hurwitz)"):
    k = sp.symbols('k', real=True)
    CE_expr = sp.expand(P_den_sym + k * P_num_sym)
    CE_poly = sp.Poly(CE_expr, s)
    coeffs_routh = CE_poly.all_coeffs()
    n_routh = CE_poly.degree()

    st.markdown("### Cruzamento com o Eixo Imaginário")
    st.markdown(r"**Equação Característica $1 + k \frac{N(s)}{D(s)} = 0 \implies D(s) + kN(s) = 0$:**")
    st.latex(rf"{sp.latex(CE_expr)} = 0")

    # Build Routh table
    routh_table = []
    row0 = coeffs_routh[0::2]
    row1 = coeffs_routh[1::2]
    max_len = max(len(row0), len(row1))
    row0.extend([0] * (max_len - len(row0)))
    row1.extend([0] * (max_len - len(row1)))
    routh_table.append(row0)
    routh_table.append(row1)

    for i in range(2, n_routh + 1):
        prev_row = routh_table[i-1]
        pprev_row = routh_table[i-2]
        new_row = []
        for j in range(len(prev_row) - 1):
            if prev_row[0] == 0:
                elem = 0
            else:
                elem = (prev_row[0] * pprev_row[j+1] - pprev_row[0] * prev_row[j+1]) / prev_row[0]
            new_row.append(sp.cancel(elem))
        new_row.append(0)
        routh_table.append(new_row)

    # Display Routh table as markdown
    table_md = "| Linha | " + " | ".join([f"Coluna {i+1}" for i in range(max_len)]) + " |\n"
    table_md += "|" + "---|" * (max_len + 1) + "\n"
    for i, row in enumerate(routh_table):
        power = n_routh - i
        row_str = f"| $s^{power}$ | "
        row_str += " | ".join([f"${sp.latex(sp.cancel(elem))}$" if str(elem) != "0" else "$0$" for elem in row]) + " |\n"
        table_md += row_str

    st.markdown("#### Tabela de Routh-Hurwitz:")
    st.markdown(table_md)

    # Find stability margin
    s1_elem = routh_table[n_routh-1][0]
    st.markdown(r"**Para encontrar a margem de estabilidade, forçamos o primeiro termo da linha $s^1$ a ser zero:**")
    st.latex(rf"{sp.latex(sp.cancel(s1_elem))} = 0")

    k_crits = []
    crossing_points = []

    if s1_elem.has(k):
        num_rh, den_rh = sp.fraction(sp.cancel(s1_elem))
        possible_ks = sp.solve(num_rh, k)
        for pk in possible_ks:
            if pk.is_real and pk > 0:
                k_crits.append(float(pk))

    for kc in k_crits:
        aux_row = routh_table[n_routh-2]
        A = aux_row[0].subs(k, kc)
        B = aux_row[1].subs(k, kc)
        aux_eq = sp.simplify(A * s**2 + B)
        st.markdown(rf"**Para o ganho crítico $k = {kc:.4f}$, a equação auxiliar (da linha $s^2$) é:**")
        st.latex(rf"{sp.latex(aux_eq)} = 0")
        roots_aux = sp.solve(aux_eq, s)
        for r in roots_aux:
            crossing_points.append(complex(r))

    valid_crossings = sorted(
        list(set([complex(pt) for pt in crossing_points if abs(pt.real) < 1e-5])),
        key=lambda x: x.imag
    )

    if valid_crossings:
        cross_str = ", ".join([f"{pt.imag:.4f}j" if pt.imag >= 0 else f"- {abs(pt.imag):.4f}j" for pt in valid_crossings])
        st.success(rf"**Pontos de cruzamento exatos no eixo imaginário:** $s = {cross_str}$")
    else:
        st.info("**O Lugar das Raízes não cruza o eixo imaginário para $k > 0$.**")

    # Plot: imaginary axis crossings
    fig9, ax9 = plt.subplots(figsize=(15, 8))
    plot_base_lgr(ax9, all_poles, all_zeros, rl_segments, all_roots)
    if Na > 0:
        line_length = 30
        for i, angle in enumerate(angles_rad):
            dx = line_length * np.cos(angle)
            dy = line_length * np.sin(angle)
            ax9.plot([sigma_A, sigma_A + dx], [0, dy], '--', color='darkorange',
                     alpha=0.5, linewidth=2, label='Assíntotas' if i == 0 else "")
    if valid_break_points:
        ax9.plot(valid_break_points, [0]*len(valid_break_points), 'd', markersize=10,
                 color='magenta', alpha=0.7)
    if valid_crossings:
        for pt in valid_crossings:
            ax9.plot(0, pt.imag, 's', markersize=12, color='cyan', markeredgecolor='navy',
                     markeredgewidth=2, label='Cruzamento Eixo Imag.' if pt == valid_crossings[0] else "")
            ax9.annotate(f'  jω = {pt.imag:.2f}', xy=(0, pt.imag), fontsize=10,
                         fontweight='bold', color='navy')
    setup_lgr_axes(ax9, all_poles, all_zeros, rl_segments, title='LGR com Cruzamentos no Eixo Imaginário')
    plt.tight_layout()
    st.pyplot(fig9)

# --- Passo 10: Ângulos de Partida e Chegada ---
with st.expander("**Passo 10:** Ângulos de Partida e Chegada"):
    st.markdown("### Ângulos de Partida (dos polos complexos) e Chegada (nos zeros complexos)")

    tol_10 = 1e-5
    complex_poles_10 = [p for p in all_poles if abs(p.imag) > tol_10]
    complex_zeros_10 = [z for z in all_zeros if abs(z.imag) > tol_10]

    departure_angles = {}
    arrival_angles = {}

    if not complex_poles_10 and not complex_zeros_10:
        st.info("**Não há polos ou zeros complexos neste sistema.** O Passo 10 não se aplica.")
    else:
        # --- Departure angles ---
        if complex_poles_10:
            st.markdown("#### Ângulos de Partida ($\\theta_d$ — Saída dos polos complexos)")
            st.markdown(r"**Fórmula:** $\theta_d = 180° - \sum \theta_i + \sum \phi_j$")

            for pk in complex_poles_10:
                angles_from_other_poles = [np.degrees(np.angle(pk - p)) for p in all_poles if not np.isclose(pk, p)]
                angles_from_zeros = [np.degrees(np.angle(pk - z)) for z in all_zeros]

                sum_theta = sum(angles_from_other_poles)
                sum_phi = sum(angles_from_zeros)

                angle_dep = (180.0 - sum_theta + sum_phi) % 360.0
                if angle_dep > 180:
                    angle_dep -= 360

                departure_angles[pk] = angle_dep

                # Show detailed calculation
                st.markdown(rf"**Polo $p = {pk.real:.4f}{pk.imag:+.4f}j$:**")
                for i, (p, ang) in enumerate(zip([p for p in all_poles if not np.isclose(pk, p)], angles_from_other_poles)):
                    st.markdown(rf"- Ângulo do polo $p = {p.real:.4f}{p.imag:+.4f}j$: $\theta_{{{i+1}}} = {ang:.2f}°$")
                for j, (z, ang) in enumerate(zip(all_zeros, angles_from_zeros)):
                    st.markdown(rf"- Ângulo do zero $z = {z.real:.4f}{z.imag:+.4f}j$: $\phi_{{{j+1}}} = {ang:.2f}°$")
                st.markdown(rf"$\sum \theta_i = {sum_theta:.2f}°$, $\sum \phi_j = {sum_phi:.2f}°$")
                st.success(rf"$\theta_d = 180° - {sum_theta:.2f}° + {sum_phi:.2f}° = {angle_dep:.2f}°$")
        else:
            st.info("Não há polos complexos — ângulos de partida não são aplicáveis.")

        # --- Arrival angles ---
        if complex_zeros_10:
            st.markdown("#### Ângulos de Chegada ($\\theta_a$ — Entrada nos zeros complexos)")
            st.markdown(r"**Fórmula:** $\theta_a = 180° - \sum \phi_i + \sum \theta_j$")

            for zk in complex_zeros_10:
                angles_from_other_zeros = [np.degrees(np.angle(zk - z)) for z in all_zeros if not np.isclose(zk, z)]
                angles_from_poles = [np.degrees(np.angle(zk - p)) for p in all_poles]

                sum_phi_z = sum(angles_from_other_zeros)
                sum_theta_z = sum(angles_from_poles)

                angle_arr = (180.0 - sum_phi_z + sum_theta_z) % 360.0
                if angle_arr > 180:
                    angle_arr -= 360

                arrival_angles[zk] = angle_arr

                # Show detailed calculation
                st.markdown(rf"**Zero $z = {zk.real:.4f}{zk.imag:+.4f}j$:**")
                for i, (p, ang) in enumerate(zip(all_poles, angles_from_poles)):
                    st.markdown(rf"- Ângulo do polo $p = {p.real:.4f}{p.imag:+.4f}j$: $\theta_{{{i+1}}} = {ang:.2f}°$")
                for j, (z, ang) in enumerate(zip([z for z in all_zeros if not np.isclose(zk, z)], angles_from_other_zeros)):
                    st.markdown(rf"- Ângulo do zero $z = {z.real:.4f}{z.imag:+.4f}j$: $\phi_{{{j+1}}} = {ang:.2f}°$")
                st.markdown(rf"$\sum \phi_i = {sum_phi_z:.2f}°$, $\sum \theta_j = {sum_theta_z:.2f}°$")
                st.success(rf"$\theta_a = 180° - {sum_phi_z:.2f}° + {sum_theta_z:.2f}° = {angle_arr:.2f}°$")
        else:
            st.info("Não há zeros complexos — ângulos de chegada não são aplicáveis.")

    # --- Plot ---
    fig10, ax10 = plt.subplots(figsize=(15, 8))
    plot_base_lgr(ax10, all_poles, all_zeros, rl_segments, all_roots)
    if Na > 0:
        line_length = 40
        for i, angle in enumerate(angles_rad):
            dx = line_length * np.cos(angle)
            dy = line_length * np.sin(angle)
            ax10.plot([sigma_A, sigma_A + dx], [0, dy], '--', color='darkorange',
                      alpha=0.4, linewidth=2, label='Assíntotas' if i == 0 else "")
    if valid_break_points:
        ax10.plot(valid_break_points, [0]*len(valid_break_points), 'd', markersize=12,
                  color='magenta', markeredgewidth=2, label='Pontos Saída/Entrada')
    if valid_crossings:
        cross_y = [pt.imag for pt in valid_crossings]
        ax10.plot([0]*len(valid_crossings), cross_y, '*', markersize=18, color='cyan',
                  markeredgewidth=2, markeredgecolor='black', label='Cruzamento jω')

    arrow_len = 1.5
    text_offset = 0.8
    extra_x = [p.real for p in all_poles] + [z.real for z in all_zeros]
    extra_y = [p.imag for p in all_poles] + [z.imag for z in all_zeros]
    for s_seg, e_seg in rl_segments:
        extra_x.extend([s_seg, e_seg])
    if Na > 0:
        extra_x.append(sigma_A)
    extra_x.extend(valid_break_points)

    # Draw departure arrows (red, outward from pole)
    for pk, angle_deg in departure_angles.items():
        angle_rad_d = np.radians(angle_deg)
        dx = arrow_len * np.cos(angle_rad_d)
        dy = arrow_len * np.sin(angle_rad_d)
        ax10.annotate('', xy=(pk.real + dx, pk.imag + dy),
                      xytext=(pk.real, pk.imag),
                      arrowprops=dict(arrowstyle='->', color='darkred', lw=2, mutation_scale=15))
        tx = pk.real + (arrow_len + text_offset) * np.cos(angle_rad_d)
        ty = pk.imag + (arrow_len + text_offset) * np.sin(angle_rad_d)
        if abs(np.sin(angle_rad_d)) < 0.3:
            ty += 0.6 * np.sign(pk.imag)
        ax10.text(tx, ty, f'{angle_deg:.1f}°', color='darkred', fontsize=10, fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='darkred',
                            boxstyle='round,pad=0.2', alpha=0.9))
        extra_x.append(pk.real + (arrow_len + text_offset + 1) * np.cos(angle_rad_d))
        extra_y.append(pk.imag + (arrow_len + text_offset + 1) * np.sin(angle_rad_d))

    # Draw arrival arrows (green, outward from zero)
    for zk, angle_deg in arrival_angles.items():
        angle_rad_a = np.radians(angle_deg)
        dx = arrow_len * np.cos(angle_rad_a)
        dy = arrow_len * np.sin(angle_rad_a)
        ax10.annotate('', xy=(zk.real + dx, zk.imag + dy),
                      xytext=(zk.real, zk.imag),
                      arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2, mutation_scale=15))
        tx = zk.real + (arrow_len + text_offset) * np.cos(angle_rad_a)
        ty = zk.imag + (arrow_len + text_offset) * np.sin(angle_rad_a)
        ax10.text(tx, ty, f'{angle_deg:.1f}°', color='darkgreen', fontsize=10, fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='darkgreen',
                            boxstyle='round,pad=0.2', alpha=0.9))
        extra_x.append(zk.real + (arrow_len + text_offset + 1) * np.cos(angle_rad_a))
        extra_y.append(zk.imag + (arrow_len + text_offset + 1) * np.sin(angle_rad_a))

    # Set limits to include all arrows and text
    if extra_x:
        x_min10, x_max10 = min(extra_x), max(extra_x)
        x_span10 = x_max10 - x_min10
        pad_x10 = x_span10 * 0.25 if x_span10 > 0 else 5.0
        ax10.set_xlim(x_min10 - pad_x10, max(x_max10 + pad_x10, 5))
    if extra_y:
        y_abs = [abs(y) for y in extra_y]
        y_limit10 = max(y_abs) + 4 if y_abs else 8
        ax10.set_ylim(-y_limit10, y_limit10)

    ax10.set_aspect('auto')
    ax10.set_title('Lugar das Raízes com Vetores de Partida/Chegada', fontsize=16, pad=20)
    ax10.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=14)
    ax10.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=14)
    ax10.grid(True, linestyle=':', alpha=0.7)
    handles, labels = ax10.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax10.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig10)

# --- Passo 11: Critério de Ângulo ---
with st.expander("**Passo 11:** Critério de Ângulo"):
    s_test = complex(s_test_real, s_test_imag)

    st.markdown(f"**Ponto de teste:** $s = {s_test.real} {s_test.imag:+}j$")

    theta_poles = [np.degrees(np.angle(s_test - p)) for p in all_poles]
    phi_zeros = [np.degrees(np.angle(s_test - z)) for z in all_zeros]

    sum_theta = sum(theta_poles)
    sum_phi = sum(phi_zeros)
    total_angle = sum_theta - sum_phi
    total_angle_norm = total_angle % 360.0
    is_on_lgr = (180.0 - threshold) <= total_angle_norm <= (180.0 + threshold)

    st.markdown(r"#### Ângulos partindo dos Polos ($\theta_i$):")
    for i, p in enumerate(all_poles):
        st.markdown(rf"Polo $p_{{{i+1}}} = {p.real:.2f}{p.imag:+.2f}j$: $\theta_{{{i+1}}} = {theta_poles[i]:.2f}°$")
    st.markdown(rf"**Somatório:** ${sum_theta:.2f}°$")

    if all_zeros:
        st.markdown(r"#### Ângulos partindo dos Zeros ($\phi_j$):")
        for j, z in enumerate(all_zeros):
            st.markdown(rf"Zero $z_{{{j+1}}} = {z.real:.2f}{z.imag:+.2f}j$: $\phi_{{{j+1}}} = {phi_zeros[j]:.2f}°$")
        st.markdown(rf"**Somatório:** ${sum_phi:.2f}°$")
    else:
        st.markdown(r"*Não há zeros, logo $\sum \phi_j = 0°$*")

    st.markdown("#### Avaliação Final:")
    st.markdown(rf"Ângulo Resultante = ${sum_theta:.2f}° - {sum_phi:.2f}° = {total_angle:.2f}°$")

    if total_angle < 0 or total_angle >= 360:
        st.markdown(rf"Normalizado (módulo 360°): ${total_angle_norm:.2f}°$")

    if is_on_lgr:
        st.success(rf"✅ O ponto **PERTENCE** ao LGR! ({total_angle_norm:.2f}° está dentro de 180° ± {threshold}°)")
    else:
        st.error(rf"❌ O ponto **NÃO PERTENCE** ao LGR! ({total_angle_norm:.2f}° fora de 180° ± {threshold}°)")

    # Plot do critério de ângulo
    fig11, ax11 = plt.subplots(figsize=(12, 6))
    ax11.plot(np.real(all_poles), np.imag(all_poles), 'x', markersize=12, color='red',
              markeredgewidth=3, label='Polos')
    if all_zeros:
        ax11.plot(np.real(all_zeros), np.imag(all_zeros), 'o', markersize=10, color='green',
                  fillstyle='none', markeredgewidth=2, label='Zeros')

    for i, (start, end) in enumerate(rl_segments):
        ax11.plot([start, end], [0, 0], color='blue', linewidth=4,
                  solid_capstyle='round', alpha=0.5)

    point_color = 'limegreen' if is_on_lgr else 'red'
    point_marker = '*' if is_on_lgr else 'X'
    point_label = f's = {s_test.real}{s_test.imag:+}j ({"Pertence" if is_on_lgr else "Não pertence"})'
    ax11.plot(s_test.real, s_test.imag, point_marker, markersize=18, color=point_color,
              markeredgecolor='black', label=point_label)

    for p in all_poles:
        ax11.plot([p.real, s_test.real], [p.imag, s_test.imag], ':', color='red', alpha=0.3)
    for z in all_zeros:
        ax11.plot([z.real, s_test.real], [z.imag, s_test.imag], ':', color='green', alpha=0.3)

    ax11.axhline(0, color='black', linewidth=1.2)
    ax11.axvline(0, color='black', linewidth=1.2)

    all_x = [p.real for p in all_poles] + [z.real for z in all_zeros] + [s_test.real]
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        ax11.set_xlim(x_min - 2, x_max + 2)
        y_coords = [abs(p.imag) for p in all_poles] + [abs(z.imag) for z in all_zeros] + [abs(s_test.imag)]
        y_limit = max(y_coords) + 2 if y_coords else 4
        ax11.set_ylim(-y_limit, y_limit)

    ax11.set_aspect('auto')
    ax11.set_title('Critério de Ângulo: Verificação de Ponto', fontsize=14)
    ax11.set_xlabel(r'Eixo Real ($\sigma$)', fontsize=12)
    ax11.set_ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=12)
    ax11.grid(True, linestyle=':', alpha=0.6)
    ax11.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig11)

# --- Passo 12: Critério de Módulo (Cálculo de K) ---
with st.expander("**Passo 12:** Critério de Módulo (Cálculo de K)"):
    st.markdown("### Critério de Módulo")
    st.markdown(r"Se um ponto $s_0$ pertence ao LGR, o valor de $K$ correspondente é dado por:")
    st.latex(r"K = \frac{1}{|P(s_0)|} = \frac{\prod |s_0 - p_i|}{\prod |s_0 - z_j|}")

    s_test_k = complex(s_test_real, s_test_imag)

    # Check if point is approximately on the LGR (reuse criterion from step 11)
    theta_p = [np.degrees(np.angle(s_test_k - p)) for p in all_poles]
    phi_z = [np.degrees(np.angle(s_test_k - z)) for z in all_zeros]
    angle_total = (sum(theta_p) - sum(phi_z)) % 360.0
    on_lgr_k = (180.0 - threshold) <= angle_total <= (180.0 + threshold)

    prod_poles = 1.0
    for p in all_poles:
        prod_poles *= abs(s_test_k - p)

    prod_zeros = 1.0
    for z in all_zeros:
        prod_zeros *= abs(s_test_k - z)

    st.markdown(f"**Ponto de teste:** $s_0 = {s_test_k.real} {s_test_k.imag:+}j$")

    st.markdown("#### Distâncias dos polos:")
    for i, p in enumerate(all_poles):
        dist = abs(s_test_k - p)
        st.markdown(rf"$|s_0 - p_{{{i+1}}}| = |{s_test_k.real}{s_test_k.imag:+}j - ({p.real:.2f}{p.imag:+.2f}j)| = {dist:.4f}$")
    st.markdown(rf"$\prod |s_0 - p_i| = {prod_poles:.4f}$")

    if all_zeros:
        st.markdown("#### Distâncias dos zeros:")
        for j, z in enumerate(all_zeros):
            dist = abs(s_test_k - z)
            st.markdown(rf"$|s_0 - z_{{{j+1}}}| = |{s_test_k.real}{s_test_k.imag:+}j - ({z.real:.2f}{z.imag:+.2f}j)| = {dist:.4f}$")
        st.markdown(rf"$\prod |s_0 - z_j| = {prod_zeros:.4f}$")
    else:
        st.markdown(r"*Não há zeros finitos, logo $\prod |s_0 - z_j| = 1$*")

    if prod_zeros > 1e-10:
        K_value = prod_poles / prod_zeros
        st.markdown("#### Resultado:")
        st.latex(rf"K = \frac{{{prod_poles:.4f}}}{{{prod_zeros:.4f}}} = {K_value:.4f}")

        if on_lgr_k:
            st.success(rf"✅ O ponto pertence ao LGR. O valor de $K$ correspondente é: **K = {K_value:.4f}**")
        else:
            st.warning(rf"⚠️ O ponto **não pertence** ao LGR (falha no critério de ângulo). O valor calculado de K = {K_value:.4f} é apenas uma referência.")
    else:
        st.error("❌ Não é possível calcular K: o ponto coincide com um zero.")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("**Calculadora LGR** - Ferramenta didática para análise de sistemas de controle.")
