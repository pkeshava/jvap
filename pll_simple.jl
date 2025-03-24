using CSV, DataFrames, Plots

# Discrete-time PLL simulator with improved PFD logic
struct PLL
    dt::Float64    # Time step (arbitrary time unit)
    Tsim::Float64  # Simulation duration (arbitrary time unit)
    Fref::Float64  # Normalized reference frequency
    slope_up::Int  # PFD UP slope (dt units)
    slope_down::Int # PFD DOWN slope (dt units)
    reset_delay::Int # PFD reset delay (dt units)
    gain_cp::Float64  # Charge pump gain (normalized)
    gain_vco::Float64 # VCO gain (normalized)
    F0::Float64    # VCO free-running frequency (normalized)
    div_N::Int     # Divider ratio
    div_rise::Int  # Divider rise time (dt units)
    div_fall::Int  # Divider fall time (dt units)
end

function simulate_pll(pll::PLL)
    dt, Tsim = pll.dt, pll.Tsim
    time_steps = Int(Tsim / dt)
    
    # Initialize variables
    phi_ref, phi_div = 0.0, 0.0
    up, down = 0.0, 0.0
    vctrl, freq_out = 0.5, pll.F0  # Start control voltage at mid-range
    output_wave = 0.0
    div_state = 0  # 0: Low, 1: High
    reset_counter = 0
    
    # Store data
    results = DataFrame(time=[], phi_ref=[], phi_div=[], up=[], down=[], vctrl=[], freq_out=[], div_output=[])

    # Simulation loop
    for step in 1:time_steps
        t = step * dt

        # Reference phase update
        phi_ref += 2π * pll.Fref * dt

        # Divider phase update
        phi_div += 2π * freq_out * dt

        # PFD Logic: Direct Phase Comparison
        if phi_ref > phi_div + 1e-6  # Reference leads
            up = 1.0
            down = 0.0
            reset_counter = pll.reset_delay
        elseif phi_div > phi_ref + 1e-6  # Divider leads
            up = 0.0
            down = 1.0
            reset_counter = pll.reset_delay
        end

        # Apply Reset Delay
        if reset_counter > 0
            reset_counter -= 1
        else
            up, down = 0.0, 0.0
        end

        # Charge Pump Output (Normalized)
        Icp_out = pll.gain_cp * (up - down)  # Normalized current-like behavior

        # Loop Filter (Simple Discrete-Time Integration)
        vctrl += Icp_out * dt  # Normalized accumulation effect

        # Limit control voltage within 0 to 1 range (simulating real-world saturation)
        vctrl = min(max(vctrl, 0.0), 1.0)

        # VCO Frequency Update (Normalized)
        freq_out = pll.F0 + pll.gain_vco * (vctrl - 0.5)  # Centered around mid-point

        # Divider logic (50% duty cycle)
        if mod(step, pll.div_N) == 0
            div_state = 1 - div_state  # Toggle state
        end

        # Simulated rise and fall times for the divider
        if div_state == 1
            output_wave = min(output_wave + dt / (pll.div_rise * dt), 1.0)
        else
            output_wave = max(output_wave - dt / (pll.div_fall * dt), 0.0)
        end

        # Store results
        push!(results, (t, phi_ref, phi_div, up, down, vctrl, freq_out, output_wave))
    end

    # Save results
    CSV.write("pll_simulation_fixed.csv", results)
    
    return results
end

# Define PLL parameters (normalized)
pll = PLL(
    1e-3, 1.0, 0.1,  # dt, Tsim, Fref
    2, 2, 2,         # slope_up, slope_down, reset_delay
    0.5, 0.8, 1.0,   # gain_cp, gain_vco, F0
    10, 2, 2         # div_N, div_rise, div_fall
)

# Run simulation
results = simulate_pll(pll)


unicodeplots()

using CSV, DataFrames, Plots
filename="pll_simulation_fixed.csv"
# Function to read PLL simulation data and plot individual signals
# Load the CSV file
df = CSV.read(filename, DataFrame)

# Plot PFD Signals
plot(df.time, df.up, label="UP", xlabel="Time (s)", ylabel="PFD Output", title="PFD Signals", linewidth=1.5)
plot!(df.time, df.down, label="DOWN", linestyle=:dash, linewidth=1.5)
savefig("pfd_signals.png")

# Plot Control Voltage (vctrl)
plot(df.time, df.vctrl, label="Control Voltage", xlabel="Time (s)", ylabel="Voltage (V)", title="Loop Filter Output (Control Voltage)", linewidth=1.5)
savefig("control_voltage.png")

# Plot VCO Frequency
plot(df.time, df.freq_out, label="VCO Frequency", xlabel="Time (s)", ylabel="Frequency (Hz)", title="VCO Frequency Response", linewidth=1.5)
savefig("vco_frequency.png")

# Plot Reference and Divider Phase
plot(df.time, df.phi_ref, label="Reference Phase", xlabel="Time (s)", ylabel="Phase (radians)", title="Reference and Divider Phase", linewidth=1.5)
plot!(df.time, df.phi_div, label="Divider Phase", linestyle=:dash, linewidth=1.5)
savefig("phase_comparison.png")

# Plot Divider Output
plot(df.time, df.div_output, label="Divider Output", xlabel="Time (s)", ylabel="State", title="Divider Output Signal", linewidth=1.5)
savefig("divider_output.png")

println("Plots saved as images (pfd_signals.png, control_voltage.png, vco_frequency.png, phase_comparison.png, divider_output.png)")

# Usage example
# plot_pll_results("pll_simulation.csv")
