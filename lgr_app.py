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


def compute_numerical_root_locus(num, den, k_max=10000, k_points=10000):
    """Compute numerical root locus for a range of K values."""
    K_vals = np.linspace(0, k_max, k_points)
    all_roots = []
    for K in K_vals:
        eq = np.polyadd(den, K * num)
        roots = np.roots(eq)
        all_roots.append(roots)
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

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("**Calculadora LGR** - Ferramenta didática para análise de sistemas de controle.")
