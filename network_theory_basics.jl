# Pouyan Keshavarzian, Ph.D.
# File created on 2025-03-06
# Last updated on 2025-03-06


# ---------------- Network Theory Basics ----------------

using ControlSystems
using Plots
s = tf("s")  # Define Laplace variable
ω_c = 2*pi*10e9     # Cutoff frequency
H = 1 / (s + ω_c)  # First-order low-pass filter
bodeplot(H)


using Symbolics
@variables s
H_lp_sym = 1 / (s + ω_c)

using Latexify
latex_equation = latexify(H_lp_sym)

@variables s ω_c
H_hp = s / (s + ω_c)  # High-pass filter transfer function

latex_equation = latexify(H_hp)
println(latex_equation)

# ---------------


using Symbolics, Latexify, ControlSystems

@variables R C s
H_RC = 1 / (1 + s * R * C)  # Symbolic transfer function
latexify(H_RC)  # Convert to LaTeX for visualization


function rc_lpf_bode(R_val, C_val)
    s = tf("s")  # Define Laplace variable
    H_numeric = 1 / (1 + s * R_val * C_val)  # Numeric transfer function
    bodeplot(H_numeric)  # Generate Bode plot
end

# Example: Set R = 1kΩ, C = 1µF
rc_lpf_bode(1e3, 1e-6)

using Plots

function plot_multiple_rc_bode(R_values, C_values)
    s = tf("s")
    p = plot(title="Bode Plot of RC Low-Pass Filter", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")

    for (R, C) in zip(R_values, C_values)
        H = 1 / (1 + s * R * C)
        bodeplot!(H, label="R=$(R)Ω, C=$(C)F")
    end

    display(p)
end

# Example: Sweep different RC values
plot_multiple_rc_bode([1e3, 10e3], [1e-6, 1e-7])