# Corrected PLL Simulator in Julia
# Simulates a Phase-Locked Loop with accurate dynamics

using Plots
using CSV
using DataFrames
using Printf

# Define component structures with realistic behavior
mutable struct PFD
    state::Symbol       # Current state: :idle, :up, :down, :both
    last_ref::Bool      # Last reference edge
    last_fb::Bool       # Last feedback edge
    reset_counter::Int  # Counter for reset delay
    reset_time::Float64 # Reset time in seconds
    dt::Float64         # Time step in seconds
end

mutable struct ChargePump
    i_up::Float64       # UP current in amperes
    i_down::Float64     # DOWN current in amperes
    current::Float64    # Current output
end

mutable struct LoopFilter
    r1::Float64         # Series resistor (ohms)
    c1::Float64         # Series capacitor (farads)
    c2::Float64         # Shunt capacitor (farads)
    v_c1::Float64       # Voltage across C1
    v_c2::Float64       # Voltage across C2
    v_out::Float64      # Output voltage
end

mutable struct VCO
    f_center::Float64   # Center frequency in Hz
    sensitivity::Float64 # Sensitivity in Hz/V
    phase::Float64      # Current phase in radians
    freq::Float64       # Current frequency in Hz
    last_phase::Float64 # Previous phase
end

mutable struct Divider
    divide_ratio::Int   # Divide ratio N
    phase::Float64      # Current phase
    last_phase::Float64 # Previous phase
end

mutable struct PLL
    pfd::PFD
    charge_pump::ChargePump
    loop_filter::LoopFilter
    vco::VCO
    divider::Divider
    dt::Float64         # Time step in seconds
    time::Float64       # Current simulation time
    ref_freq::Float64   # Reference frequency in Hz
    ref_phase::Float64  # Reference phase in radians
    last_ref_phase::Float64 # Previous reference phase
    
    # For analysis
    natural_frequency::Float64  # Natural frequency in rad/s
    damping_factor::Float64     # Damping factor
    
    # Historical data for plotting
    time_history::Vector{Float64}
    control_voltage_history::Vector{Float64}
    vco_freq_history::Vector{Float64}
    ref_phase_history::Vector{Float64}
    div_phase_history::Vector{Float64}
    pfd_up_history::Vector{Bool}
    pfd_down_history::Vector{Bool}
    cp_current_history::Vector{Float64}
end

# Constructor functions
function create_pfd(reset_time::Float64, dt::Float64)
    PFD(:idle, false, false, 0, reset_time, dt)
end

function create_charge_pump(i_up::Float64, i_down::Float64)
    ChargePump(i_up, i_down, 0.0)
end

function create_loop_filter(r1::Float64, c1::Float64, c2::Float64)
    LoopFilter(r1, c1, c2, 0.0, 0.0, 0.0)
end

function create_vco(f_center::Float64, sensitivity::Float64)
    VCO(f_center, sensitivity, 0.0, f_center, 0.0)
end

function create_divider(divide_ratio::Int)
    Divider(divide_ratio, 0.0, 0.0)
end

function create_pll(
    reset_time::Float64,
    i_up::Float64, 
    i_down::Float64, 
    r1::Float64, 
    c1::Float64, 
    c2::Float64, 
    f_center::Float64, 
    sensitivity::Float64, 
    divide_ratio::Int, 
    dt::Float64, 
    ref_freq::Float64,
    initial_vco_offset::Float64 = 0.0  # Optional initial frequency offset in Hz
)
    # Create components
    pfd = create_pfd(reset_time, dt)
    charge_pump = create_charge_pump(i_up, i_down)
    loop_filter = create_loop_filter(r1, c1, c2)
    vco = create_vco(f_center, sensitivity)
    divider = create_divider(divide_ratio)
    
    # Apply initial frequency offset if specified
    if initial_vco_offset != 0.0
        vco.freq = f_center + initial_vco_offset
    end
    
    # Calculate loop parameters
    cp_gain = i_up / (2π)  # A/rad
    vco_gain = sensitivity / (2π)  # Hz/V/rad = rad/s/V/rad = rad/s/V
    n = divide_ratio
    
    # Natural frequency (rad/s)
    ωn = sqrt(cp_gain * vco_gain / (n * (c1 + c2)))
    
    # Damping factor
    ζ = 0.5 * r1 * c1 * ωn
    
    PLL(
        pfd,
        charge_pump,
        loop_filter,
        vco,
        divider,
        dt,
        0.0,   # initial time
        ref_freq,
        0.0,   # initial ref phase
        0.0,   # initial last ref phase
        ωn,    # natural frequency
        ζ,     # damping factor
        Float64[], Float64[], Float64[], Float64[], Float64[], Bool[], Bool[], Float64[]
    )
end

# Update functions for each component
function update_pfd!(pfd::PFD, ref_phase::Float64, last_ref_phase::Float64, div_phase::Float64, last_div_phase::Float64)
    # Detect reference phase wrapping (rising edge)
    ref_edge = (last_ref_phase > 3π/2 && ref_phase < π/2)
    
    # Detect divider phase wrapping (rising edge)
    div_edge = (last_div_phase > 3π/2 && div_phase < π/2)
    
    # Default outputs
    up_out = false
    down_out = false
    
    # State machine 
    if pfd.reset_counter > 0
        # In reset state
        pfd.reset_counter -= 1
        # Keep state, but outputs stay at 0
    else
        # Normal operation
        if ref_edge && div_edge
            # Both edges simultaneously - rare but possible
            pfd.state = :both
            pfd.reset_counter = round(Int, pfd.reset_time / pfd.dt)
        elseif ref_edge
            # Reference edge detected
            if pfd.state == :idle || pfd.state == :down
                pfd.state = :up
            elseif pfd.state == :up
                # Already in UP state, do nothing
            end
        elseif div_edge
            # Divider edge detected
            if pfd.state == :idle || pfd.state == :up
                pfd.state = :down
            elseif pfd.state == :down
                # Already in DOWN state, do nothing
            end
        end
        
        # Cross triggering (both signals active)
        if pfd.state == :up && div_edge
            pfd.state = :both
            pfd.reset_counter = round(Int, pfd.reset_time / pfd.dt)
        elseif pfd.state == :down && ref_edge
            pfd.state = :both
            pfd.reset_counter = round(Int, pfd.reset_time / pfd.dt)
        end
    end
    
    # Set outputs based on state
    if pfd.state == :up
        up_out = true
    elseif pfd.state == :down
        down_out = true
    end
    
    # Update last edge values
    pfd.last_ref = ref_edge
    pfd.last_fb = div_edge
    
    return up_out, down_out
end

function update_charge_pump!(cp::ChargePump, up::Bool, down::Bool)
    # Calculate net current from UP and DOWN signals
    if up && !down
        cp.current = cp.i_up
    elseif down && !up
        cp.current = -cp.i_down
    else
        cp.current = 0.0
    end
    
    return cp.current
end

function update_loop_filter!(lf::LoopFilter, current_in::Float64, dt::Float64)
    # Lead-lag filter implementation
    
    # Update voltage across C1 (series capacitor)
    dv_c1 = (current_in * dt) / lf.c1
    lf.v_c1 += dv_c1
    
    # Calculate voltage across R1
    v_r1 = current_in * lf.r1
    
    # Update voltage across C2 (parallel capacitor)
    dv_c2 = (current_in * dt) / lf.c2
    lf.v_c2 += dv_c2
    
    # Total control voltage is sum of voltages across components
    lf.v_out = lf.v_c1 + v_r1
    
    return lf.v_out
end

function update_vco!(vco::VCO, control_voltage::Float64, dt::Float64)
    # Store previous phase
    vco.last_phase = vco.phase
    
    # Update VCO frequency based on control voltage
    vco.freq = vco.f_center + vco.sensitivity * control_voltage
    
    # Limit frequency to reasonable values (prevent negative frequencies)
    if vco.freq < 0
        vco.freq = 0
    end
    
    # Update phase based on frequency
    delta_phase = 2π * vco.freq * dt
    vco.phase = (vco.phase + delta_phase) % (2π)
    
    return vco.freq, vco.phase
end

function update_divider!(div::Divider, vco_phase::Float64, vco_last_phase::Float64)
    # Store previous phase
    div.last_phase = div.phase
    
    # Detect VCO phase wrapping (rising edge)
    if vco_last_phase > 3π/2 && vco_phase < π/2
        # Increment divider phase by 2π/N
        div.phase = (div.phase + 2π / div.divide_ratio) % (2π)
    end
    
    return div.phase
end

function update_pll!(pll::PLL)
    # Store previous reference phase
    pll.last_ref_phase = pll.ref_phase
    
    # Update reference phase
    pll.ref_phase = (pll.ref_phase + 2π * pll.ref_freq * pll.dt) % (2π)
    
    # Update VCO
    control_voltage = update_loop_filter!(pll.loop_filter, pll.charge_pump.current, pll.dt)
    vco_freq, vco_phase = update_vco!(pll.vco, control_voltage, pll.dt)
    
    # Update divider
    div_phase = update_divider!(pll.divider, vco_phase, pll.vco.last_phase)
    
    # Update PFD
    up, down = update_pfd!(pll.pfd, pll.ref_phase, pll.last_ref_phase, div_phase, pll.divider.last_phase)
    
    # Update charge pump
    cp_current = update_charge_pump!(pll.charge_pump, up, down)
    
    # Update time
    pll.time += pll.dt
    
    # Record history for plotting
    push!(pll.time_history, pll.time)
    push!(pll.control_voltage_history, control_voltage)
    push!(pll.vco_freq_history, vco_freq)
    push!(pll.ref_phase_history, pll.ref_phase)
    push!(pll.div_phase_history, div_phase)
    push!(pll.pfd_up_history, up)
    push!(pll.pfd_down_history, down)
    push!(pll.cp_current_history, cp_current)
    
    return pll
end

function simulate_pll(pll::PLL, duration::Float64)
    steps = Int(ceil(duration / pll.dt))
    
    # Limit steps to prevent memory issues
    max_steps = 1000000
    if steps > max_steps
        @warn "Limiting simulation to $max_steps steps to prevent memory issues"
        steps = max_steps
        duration = steps * pll.dt
    end
    
    # Pre-allocate arrays for better performance
    pll.time_history = zeros(Float64, steps)
    pll.control_voltage_history = zeros(Float64, steps)
    pll.vco_freq_history = zeros(Float64, steps)
    pll.ref_phase_history = zeros(Float64, steps)
    pll.div_phase_history = zeros(Float64, steps)
    pll.pfd_up_history = falses(steps)
    pll.pfd_down_history = falses(steps)
    pll.cp_current_history = zeros(Float64, steps)
    
    # Run simulation
    for i in 1:steps
        update_pll!(pll)
        
        # Store data in pre-allocated arrays
        pll.time_history[i] = pll.time
        pll.control_voltage_history[i] = pll.loop_filter.v_out
        pll.vco_freq_history[i] = pll.vco.freq
        pll.ref_phase_history[i] = pll.ref_phase
        pll.div_phase_history[i] = pll.divider.phase
        pll.pfd_up_history[i] = pll.charge_pump.current > 0
        pll.pfd_down_history[i] = pll.charge_pump.current < 0
        pll.cp_current_history[i] = pll.charge_pump.current
    end
    
    return pll, duration
end

# Function to save simulation data to CSV
function save_pll_data_to_csv(pll::PLL, filename::String)
    # Limit the number of points to save to prevent huge files
    max_rows = 10000
    data_length = length(pll.time_history)
    
    if data_length > max_rows
        # Sample points evenly from the full dataset
        indices = round.(Int, range(1, data_length, length=max_rows))
        
        # Create a DataFrame with sampled data
        df = DataFrame(
            time = pll.time_history[indices],
            control_voltage = pll.control_voltage_history[indices],
            vco_freq = pll.vco_freq_history[indices],
            ref_phase = pll.ref_phase_history[indices],
            div_phase = pll.div_phase_history[indices],
            pfd_up = Int.(pll.pfd_up_history[indices]),
            pfd_down = Int.(pll.pfd_down_history[indices]),
            charge_pump_current = pll.cp_current_history[indices]
        )
    else
        # Use all data if under threshold
        df = DataFrame(
            time = pll.time_history,
            control_voltage = pll.control_voltage_history,
            vco_freq = pll.vco_freq_history,
            ref_phase = pll.ref_phase_history,
            div_phase = pll.div_phase_history,
            pfd_up = Int.(pll.pfd_up_history),
            pfd_down = Int.(pll.pfd_down_history),
            charge_pump_current = pll.cp_current_history
        )
    end
    
    # Save to CSV
    CSV.write(filename, df)
    println("Data saved to $filename with $(nrow(df)) rows")
    
    return df
end

# Function to plot results
function plot_results(pll::PLL)
    # Calculate the expected frequency
    expected_freq = pll.ref_freq * pll.divider.divide_ratio
    
    # Plot control voltage
    p1 = plot(pll.time_history, pll.control_voltage_history, 
        label="Control Voltage (V)", 
        title="PLL Control Voltage",
        xlabel="Time (s)",
        ylabel="Voltage (V)",
        linewidth=2)
    
    # Plot VCO frequency
    p2 = plot(pll.time_history, pll.vco_freq_history ./ 1e6, 
        label="VCO Frequency (MHz)", 
        title="VCO Frequency",
        xlabel="Time (s)",
        ylabel="Frequency (MHz)",
        linewidth=2)
    
    # Add target frequency line
    hline!(p2, [expected_freq/1e6], label="Target", linestyle=:dash, color=:red)
    
    # Plot phase comparison
    p3 = plot(pll.time_history[1:min(end,10000)], (pll.ref_phase_history[1:min(end,10000)] ./ (2π)),
        label="Reference Phase (cycles)",
        title="Phase Comparison",
        xlabel="Time (s)",
        ylabel="Phase (cycles)",
        linewidth=1,
        color=:blue)
    plot!(p3, pll.time_history[1:min(end,10000)], (pll.div_phase_history[1:min(end,10000)] ./ (2π)),
        label="Divider Phase (cycles)",
        linewidth=1,
        color=:red)
    
    # Plot charge pump current
    p4 = plot(pll.time_history, pll.cp_current_history .* 1000, 
        label="Charge Pump Current (mA)", 
        title="Charge Pump Current",
        xlabel="Time (s)",
        ylabel="Current (mA)",
        linewidth=1)
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(4,1), size=(800,1000))
end

# Function to analyze PLL lock performance
function analyze_lock_performance(pll::PLL)
    # Expected final frequency
    expected_freq = pll.ref_freq * pll.divider.divide_ratio
    
    # Calculate error signal
    freq_error = (pll.vco_freq_history .- expected_freq) ./ expected_freq * 100
    
    # Find time to lock within different error thresholds
    thresholds = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
    lock_times = Float64[]
    
    for threshold in thresholds
        idx = findfirst(abs.(freq_error) .<= threshold)
        if idx !== nothing
            push!(lock_times, pll.time_history[idx])
        else
            push!(lock_times, NaN)
        end
    end
    
    # Create plot of frequency error over time
    p1 = plot(pll.time_history, freq_error, 
        label="Frequency Error (%)", 
        title="PLL Frequency Error",
        xlabel="Time (s)",
        ylabel="Error (%)",
        linewidth=2,
        color=:blue)
    
    # Add threshold lines
    for (i, threshold) in enumerate(thresholds)
        if !isnan(lock_times[i])
            hline!(p1, [threshold], linestyle=:dash, color=:gray, label="")
            hline!(p1, [-threshold], linestyle=:dash, color=:gray, label="")
            scatter!(p1, [lock_times[i]], [threshold], marker=:circle, color=:red, 
                    markersize=4, label="")
            annotate!(p1, lock_times[i], threshold*1.2, text("$(threshold)%", 8, :black))
        end
    end
    
    # Create a second plot for the lock time analysis
    p2 = bar(["±$(t)%" for t in thresholds], lock_times .* 1e6,
        title="Lock Time Analysis",
        xlabel="Error Threshold",
        ylabel="Time (μs)",
        legend=false,
        color=:blue,
        alpha=0.7)
    
    # Calculate and add theoretical settling time
    ts_theory = 4 / (pll.damping_factor * pll.natural_frequency)
    hline!(p2, [ts_theory * 1e6], linestyle=:dash, color=:red, 
           label="Theoretical (4/ζωn)")
    
    # Arrange the two plots
    plot(p1, p2, layout=(2,1), size=(800,800))
end

# Run an example simulation with parameters tuned for realistic second-order dynamics
function run_realistic_pll()
    # Create a PLL with parameters set for optimal PLL damping
    # Using a damping factor of ~0.7 (classic optimal value)
    # pll = create_pll(
    #     5e-9,            # reset_time (s) - brief reset pulse
    #     0.2e-3,          # i_up (A) - moderate charge pump current
    #     0.2e-3,          # i_down (A) - matched currents
    #     5e3,             # r1 (ohms) - sized for proper damping
    #     20e-12,          # c1 (F) - proper loop filter sizing
    #     200e-12,         # c2 (F) - 10x C1 for stability
    #     70e6,            # f_center (Hz) - offset from expected 80MHz
    #     5e6,             # sensitivity (Hz/V) - moderate sensitivity 
    #     8,               # divide_ratio
    #     1e-10,           # dt (s) - fine time step for accuracy
    #     10e6,            # ref_freq (Hz)
    #     20e6             # initial VCO offset (Hz) - start 20MHz away from lock
    # )
    #     pll = create_pll(
    #     2.5e-9,            # reset_time (s) - brief reset pulse
    #     0.1e-3,          # i_up (A) - moderate charge pump current
    #     0.1e-3,          # i_down (A) - matched currents
    #     5e3,             # r1 (ohms) - sized for proper damping
    #     20e-12,          # c1 (F) - proper loop filter sizing
    #     200e-12,         # c2 (F) - 10x C1 for stability
    #     70e6,            # f_center (Hz) - offset from expected 80MHz
    #     4e6,             # sensitivity (Hz/V) - moderate sensitivity 
    #     8,               # divide_ratio
    #     1e-10,           # dt (s) - fine time step for accuracy
    #     10e6,            # ref_freq (Hz)
    #     20e6             # initial VCO offset (Hz) - start 20MHz away from lock
    # )
    pll = create_pll(
        1.0e-9,          # reset_time (s) - brief reset pulse
        0.05e-3,          # i_up (A) - moderate charge pump current
        0.05e-3,          # i_down (A) - matched currents
        10e3,             # r1 (ohms) - sized for proper damping
        25e-12,          # c1 (F) - proper loop filter sizing
        200e-12,         # c2 (F) - 10x C1 for stability
        70e6,            # f_center (Hz) - offset from expected 80MHz
        2e6,             # sensitivity (Hz/V) - moderate sensitivity 
        8,               # divide_ratio
        1e-10,           # dt (s) - fine time step for accuracy
        10e6,            # ref_freq (Hz)
        20e6             # initial VCO offset (Hz) - start 20MHz away from lock
    )
    # Print theoretical parameters
    println("PLL Loop Parameters:")
    @printf("Natural frequency (ωn): %.2f Hz = %.2f rad/s\n", pll.natural_frequency/(2π), pll.natural_frequency)
    @printf("Damping factor (ζ): %.2f\n", pll.damping_factor)
    
    # Calculate theoretical settling time (4 time constants)
    settling_time = 4.0 / (pll.damping_factor * pll.natural_frequency)
    @printf("Theoretical settling time (4/ζωn): %.6f seconds\n", settling_time)
    
    # Also calculate overshoot percentage
    if pll.damping_factor < 1.0
        # For underdamped systems
        overshoot_pct = 100 * exp(-π * pll.damping_factor / sqrt(1 - pll.damping_factor^2))
        @printf("Theoretical peak overshoot: %.1f%%\n", overshoot_pct)
    else
        println("Theoretical peak overshoot: 0% (overdamped)")
    end
    
    # Run simulation for 3x the theoretical settling time to ensure we see full behavior
    simulate_duration = 3.0 * settling_time
    @printf("Running simulation for %.6f seconds\n", simulate_duration)
    
    # Run simulation
    pll, actual_duration = simulate_pll(pll, simulate_duration)
    
    # Check lock status
    expected_freq = pll.ref_freq * pll.divider.divide_ratio
    final_freq = pll.vco_freq_history[end]
    freq_error = abs(final_freq - expected_freq) / expected_freq
    locked = freq_error < 0.01  # 1% tolerance
    
    println("Simulation complete!")
    @printf("Final frequency: %.2f MHz\n", final_freq/1e6)
    @printf("Expected frequency: %.2f MHz\n", expected_freq/1e6)
    @printf("Frequency error: %.2f%%\n", freq_error*100)
    println("Lock status: $(locked ? "Locked" : "Not locked")")
    
    # Save data to CSV with reduced data points
    save_pll_data_to_csv(pll, "pll_simulation_data.csv")
    
    # Plot results
    p1 = plot_results(pll)
    #savefig(p1, "pll_results.png")
    #display(p1)
    
    # Analyze lock performance
    p2 = analyze_lock_performance(pll)
    #savefig(p2, "pll_lock_analysis.png")
    #display(p2)
    
    return pll
end

# Run the realistic PLL simulation when executing this file
pll=run_realistic_pll()
#p1=plot_results(pll)
#p2=analyze_lock_performance(pll)


df = CSV.File("pll_simulation_data.csv"; select=[:vco_freq]) |> DataFrame
timestep = 1e-9  # 1 ns
x = timestep * (0:(size(df, 1) - 1))  # Generate x values
y = df.vco_freq  # Extract the vco_freq column
plot(x, y, xlabel="Time (s)", ylabel="VCO Frequency (Hz)", title="VCO Frequency vs. Time", lw=2)

df = CSV.File("pll_simulation_data.csv"; select=[:control_voltage]) |> DataFrame
timestep = 1e-9  # 1 ns
x = timestep * (0:(size(df, 1) - 1))  # Generate x values
y = df.control_voltage  # Extract the vco_freq column
plot(x, y, xlabel="Time (s)", ylabel="Control Volts", title="VCO Frequency vs. Time", lw=2)

