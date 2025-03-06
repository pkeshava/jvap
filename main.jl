include("./src/jvap.jl")
using .jvap

# Example usage
word = "cross"
if is_verilogams_keyword(word)
    println("$word is a Verilog-AMS keyword in category: ", get_keyword_category(word))
else
    println("$word is not a Verilog-AMS keyword")
end



using Plots

# Define a type for piecewise-linear signals.
mutable struct PWLSignal
    time::Float64       # Last update time
    value::Float64      # Signal value at that time
    slope::Float64      # Slope of the signal (dV/dt)
    next_event::Float64 # Next scheduled update time
end

# Define a type for simulation events.
mutable struct Event
    time::Float64       # When the event should occur
    callback::Function  # Function to call at the event time
end

# Our event queue will be a simple array of events.
event_queue = Event[]

# Helper function to schedule an event.
function schedule_event(event::Event)
    push!(event_queue, event)
end

# Helper function to pop the event with the smallest time using findmin.
function pop_next_event()
    if isempty(event_queue)
        return nothing
    end
    times = [e.time for e in event_queue]
    _, idx = findmin(times)
    event = event_queue[idx]
    deleteat!(event_queue, idx)
    return event
end

# Simulation parameters
const T_max = 0.1   # total simulation time in seconds (shorter for clearer plots)
const dt = 0.001    # fixed event time step (1 ms)

# Circuit parameters
const frequency = 50.0       # 50 Hz sine wave
const amplitude = 1.0        # amplitude of the sine source
const R = 1000.0             # resistance in ohms
const C = 1e-6               # capacitance in farads
const tau = R * C            # RC time constant

# Arrays to record simulation data.
times_sine = Float64[]
values_sine = Float64[]

times_lpf = Float64[]
values_lpf = Float64[]

# Function to update the sine source signal.
function update_sine(sine::PWLSignal, current_time::Float64)
    new_value = amplitude * sin(2π * frequency * current_time)
    new_slope = amplitude * 2π * frequency * cos(2π * frequency * current_time)
    next_event = current_time + dt
    # Update the sine signal's state.
    sine.time = current_time
    sine.value = new_value
    sine.slope = new_slope
    sine.next_event = next_event
    # Record the sine signal data.
    push!(times_sine, current_time)
    push!(values_sine, new_value)
end

# Function to update the LPF output given the sine source as input.
# We compute the output at time new_time = current_time + dt to capture the delay.
function update_lpf(lpf::PWLSignal, input::PWLSignal, current_time::Float64)
    Δ = dt
    v0 = lpf.value              # previous LPF output
    a = input.value             # input value at current_time
    b = input.slope             # input slope at current_time
    exp_factor = exp(-Δ/tau)
    new_value = v0 * exp_factor + a * (1 - exp_factor) + b * (Δ - tau * (1 - exp_factor))
    new_slope = - (v0/tau) * exp_factor + (a/tau) * exp_factor + b * (1 - exp_factor)
    new_time = current_time + dt  # record LPF output at a later time
    next_event = new_time + dt
    # Update the LPF signal's state.
    lpf.time = new_time
    lpf.value = new_value
    lpf.slope = new_slope
    lpf.next_event = next_event
    # Record the LPF signal data.
    push!(times_lpf, new_time)
    push!(values_lpf, new_value)
end

# Create initial signals.
sine_signal = PWLSignal(0.0, amplitude * sin(0.0), amplitude * 2π * frequency * cos(0.0), dt)
lpf_signal = PWLSignal(0.0, 0.0, 0.0, dt)

# Define the callback for sine source updates.
function sine_callback(current_time)
    update_sine(sine_signal, current_time)
    if current_time < T_max
        schedule_event(Event(current_time + dt, sine_callback))
    end
end

# Define the callback for LPF updates.
function lpf_callback(current_time)
    update_lpf(lpf_signal, sine_signal, current_time)
    if current_time < T_max
        schedule_event(Event(current_time + dt, lpf_callback))
    end
end

# Schedule the initial events.
schedule_event(Event(0.0, sine_callback))
schedule_event(Event(0.0, lpf_callback))

# Main simulation loop.
current_time = 0.0
while !isempty(event_queue) && current_time <= T_max
    event = pop_next_event()
    if event === nothing
        break
    end
    current_time = event.time
    event.callback(current_time)
end

println("Simulation complete.")

# Compute the analytic closed-form steady-state solution for the RC low-pass filter.
# For a sine input, the steady-state solution is:
# V_out(t) = (amplitude / sqrt(1+(ωτ)^2)) * sin(2π*frequency*t - atan(ωτ))
ω = 2π * frequency
t_closed = 0:dt:T_max
analytic_lpf = [ amplitude/sqrt(1+(ω*tau)^2) * sin(ω*t - atan(ω*tau)) for t in t_closed ]

# Plot the signals:
# - "Continuous Sine" shows the ideal sine input.
# - "Simulated Sine" shows the sine values recorded during simulation.
# - "Simulated LPF" shows the LPF output from our event-driven simulation.
# - "Analytic LPF" shows the closed-form analytic solution for comparison.
plt = plot(t_closed, [amplitude*sin(ω*t) for t in t_closed],
    label="Continuous Sine",
    xlabel="Time (s)",
    ylabel="Voltage",
    title="Comparison of Simulated LPF and Analytic Closed-Form",
    lw=2)
plot!(times_sine, values_sine, label="Simulated Sine", lw=2, ls=:dash)
plot!(times_lpf, values_lpf, label="Simulated LPF", lw=2)
plot!(t_closed, analytic_lpf, label="Analytic LPF", lw=2, ls=:dot)

display(plt)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

using Plots

# --- Basic Data Structures ---

# Piecewise-linear signal representing voltage or current.
mutable struct PWLSignal
    time::Float64       # Time of last update
    value::Float64      # Signal value at that time
    slope::Float64      # Derivative (d/dt) of the signal
    next_event::Float64 # Next scheduled update time
end

# Abstract type for circuit blocks.
abstract type CircuitBlock end

# A simulation event, with a scheduled time and a callback function.
mutable struct Event
    time::Float64
    callback::Function
end

# Simple event queue implemented as an array.
event_queue = Event[]

# Schedule an event by pushing it onto the queue.
function schedule_event(event::Event)
    push!(event_queue, event)
end

# Pop the event with the smallest time.
function pop_next_event()
    if isempty(event_queue)
        return nothing
    end
    times = [e.time for e in event_queue]
    _, idx = findmin(times)
    event = event_queue[idx]
    deleteat!(event_queue, idx)
    return event
end

# --- Simulation Parameters ---
const T_max = 0.1    # Total simulation time (seconds)
const dt = 0.001     # Time step (seconds)

# Circuit parameters (common to our example)
const frequency = 50.0       # 50 Hz sine source
const amplitude = 1.0        # Sine amplitude
const R_val = 1000.0         # Resistance for RC (ohms)
const C_val = 1e-6           # Capacitance (farads)
const tau = R_val * C_val    # Time constant

# Data storage for plotting.
sine_times = Float64[]
sine_voltages = Float64[]

lpf_times = Float64[]
lpf_voltages = Float64[]

resistor_times = Float64[]
resistor_voltages = Float64[]
resistor_currents = Float64[]

# --- Modular Circuit Blocks ---

# 1. Sine Source Block: produces a voltage signal.
mutable struct SineSourceBlock <: CircuitBlock
    frequency::Float64
    amplitude::Float64
    output::PWLSignal    # Output voltage signal
end

function update!(block::SineSourceBlock, t::Float64)
    new_value = block.amplitude * sin(2π * block.frequency * t)
    new_slope = block.amplitude * 2π * block.frequency * cos(2π * block.frequency * t)
    block.output.time = t
    block.output.value = new_value
    block.output.slope = new_slope
    block.output.next_event = t + dt
    push!(sine_times, t)
    push!(sine_voltages, new_value)
end

# 2. Low Pass Filter Block: filters an input voltage using RC dynamics.
mutable struct LowPassFilterBlock <: CircuitBlock
    R::Float64
    C::Float64
    tau::Float64
    input::PWLSignal     # Input voltage (e.g. from SineSourceBlock)
    output::PWLSignal    # Filtered output voltage
end

function update!(block::LowPassFilterBlock, t::Float64)
    Δ = dt
    # v0 is the previous filter output.
    v0 = block.output.value
    # a and b represent the current input voltage and its slope.
    a = block.input.value
    b = block.input.slope
    exp_factor = exp(-Δ/block.tau)
    new_value = v0 * exp_factor + a * (1 - exp_factor) + b * (Δ - block.tau*(1 - exp_factor))
    new_slope = - (v0/block.tau)*exp_factor + (a/block.tau)*exp_factor + b*(1 - exp_factor)
    new_time = t + dt
    block.output.time = new_time
    block.output.value = new_value
    block.output.slope = new_slope
    block.output.next_event = new_time + dt
    push!(lpf_times, new_time)
    push!(lpf_voltages, new_value)
end

# 3. Resistor Load Block: represents a resistive load (impedance) and computes current.
mutable struct ResistorLoadBlock <: CircuitBlock
    R::Float64
    input::PWLSignal     # Input voltage (e.g. from LPF block)
    voltage::PWLSignal   # Voltage across resistor (assumed same as input)
    current::PWLSignal   # Computed current = voltage / R
end

function update!(block::ResistorLoadBlock, t::Float64)
    # For a resistor, current is computed instantaneously.
    v = block.input.value
    i = v / block.R
    block.voltage.time = t
    block.voltage.value = v
    block.voltage.slope = 0.0
    block.voltage.next_event = t + dt
    block.current.time = t
    block.current.value = i
    block.current.slope = 0.0
    block.current.next_event = t + dt
    push!(resistor_times, t)
    push!(resistor_voltages, v)
    push!(resistor_currents, i)
end

# --- Create Instances of Blocks and Connect Them ---

# Sine Source Block instance.
sine_output = PWLSignal(0.0, amplitude * sin(0.0), amplitude * 2π * frequency * cos(0.0), dt)
sine_block = SineSourceBlock(frequency, amplitude, sine_output)

# Low Pass Filter Block instance.
# Its input is directly connected to the sine source output.
lpf_input = sine_block.output
lpf_output = PWLSignal(0.0, 0.0, 0.0, dt)
lpf_block = LowPassFilterBlock(R_val, C_val, tau, lpf_input, lpf_output)

# Resistor Load Block instance.
# Its input is the output of the LPF.
resistor_input = lpf_block.output
resistor_voltage = PWLSignal(0.0, 0.0, 0.0, dt)
resistor_current = PWLSignal(0.0, 0.0, 0.0, dt)
resistor_block = ResistorLoadBlock(R_val, resistor_input, resistor_voltage, resistor_current)

# --- Event Callbacks for Each Block ---
function sine_callback(t)
    update!(sine_block, t)
    if t < T_max
        schedule_event(Event(t + dt, sine_callback))
    end
end

function lpf_callback(t)
    update!(lpf_block, t)
    if t < T_max
        schedule_event(Event(t + dt, lpf_callback))
    end
end

function resistor_callback(t)
    update!(resistor_block, t)
    if t < T_max
        schedule_event(Event(t + dt, resistor_callback))
    end
end

# --- Schedule Initial Events ---
schedule_event(Event(0.0, sine_callback))
schedule_event(Event(0.0, lpf_callback))
schedule_event(Event(0.0, resistor_callback))

# --- Main Simulation Loop ---
current_time = 0.0
while !isempty(event_queue) && current_time <= T_max
    event = pop_next_event()
    if event === nothing
        break
    end
    current_time = event.time
    event.callback(current_time)
end

println("Simulation complete.")

# --- Plotting the Results ---
plt1 = plot(sine_times, sine_voltages, label="Sine Source Voltage", xlabel="Time (s)", ylabel="Voltage (V)", title="Sine Source", lw=2)
plt2 = plot(lpf_times, lpf_voltages, label="LPF Output Voltage", xlabel="Time (s)", ylabel="Voltage (V)", title="Low Pass Filter", lw=2)
plt3 = plot(resistor_times, resistor_voltages, label="Resistor Voltage", xlabel="Time (s)", ylabel="Voltage (V)", title="Resistor Voltage", lw=2)
plt4 = plot(resistor_times, resistor_currents, label="Resistor Current", xlabel="Time (s)", ylabel="Current (A)", title="Resistor Current", lw=2)

plot(plt1, plt2, plt3, plt4, layout=(2,2), legend=:bottomright)
display(Plots.current())

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

using Plots
using FFTW  # Julia's standard library FFT

# ========================
#     GLOBAL SETTINGS
# ========================
const ac_mode = true   # We only do AC (impulse) analysis here
const T_max = 0.2      # total simulation time (seconds)
const dt = 0.0005      # time step (seconds)

# For better frequency resolution, you can increase T_max or reduce dt

# ========================
#  CIRCUIT PARAMETERS
# ========================
const R_val = 1000.0
const C_val = 1e-6
const tau = R_val * C_val

# ========================
#  DATA STRUCTURES
# ========================

mutable struct PWLSignal
    time::Float64
    value::Float64
    slope::Float64
    next_event::Float64
end

abstract type CircuitBlock end

mutable struct Event
    time::Float64
    callback::Function
end

event_queue = Event[]  # Our event list

function schedule_event(event::Event)
    push!(event_queue, event)
end

function pop_next_event()
    isempty(event_queue) && return nothing
    times = [e.time for e in event_queue]
    _, idx = findmin(times)
    event = event_queue[idx]
    deleteat!(event_queue, idx)
    return event
end

# ========================
#  AC (Impulse) BLOCKS
# ========================

# 1) Impulse Source Block: outputs 1.0 at t=0, then 0.0 afterwards
mutable struct ImpulseSourceBlock <: CircuitBlock
    output::PWLSignal  # voltage signal
end

function update!(block::ImpulseSourceBlock, t::Float64)
    # Impulse at t=0
    val = isapprox(t, 0.0; atol=1e-12) ? 1.0 : 0.0
    block.output.time = t
    block.output.value = val
    block.output.slope = 0.0
    block.output.next_event = t + dt
end

# 2) RC Low-Pass Filter Block
mutable struct LowPassFilterBlock <: CircuitBlock
    R::Float64
    C::Float64
    tau::Float64
    input::PWLSignal     # input voltage
    output::PWLSignal    # output voltage
end

function update!(block::LowPassFilterBlock, t::Float64)
    Δ = dt
    v0 = block.output.value
    a  = block.input.value
    b  = block.input.slope   # slope is 0 for impulse, but let's keep the formula general
    ef = exp(-Δ/block.tau)
    new_value = v0 * ef + a * (1 - ef) + b * (Δ - block.tau*(1 - ef))
    new_slope = -(v0/block.tau)*ef + (a/block.tau)*ef + b*(1 - ef)
    t_out = t + Δ
    block.output.time = t_out
    block.output.value = new_value
    block.output.slope = new_slope
    block.output.next_event = t_out + dt
end

# ========================
#  CIRCUIT INSTANCES
# ========================

# Impulse Source
impulse_signal = PWLSignal(0.0, 1.0, 0.0, dt)
impulse_block  = ImpulseSourceBlock(impulse_signal)

# LPF
lpf_input  = impulse_block.output
lpf_output = PWLSignal(0.0, 0.0, 0.0, dt)
lpf_block  = LowPassFilterBlock(R_val, C_val, tau, lpf_input, lpf_output)

# ========================
#  CALLBACKS & EVENTS
# ========================

function impulse_callback(t)
    update!(impulse_block, t)
    if t < T_max
        schedule_event(Event(t + dt, impulse_callback))
    end
end

function lpf_callback(t)
    update!(lpf_block, t)
    if t < T_max
        schedule_event(Event(t + dt, lpf_callback))
    end
end

schedule_event(Event(0.0, impulse_callback))
schedule_event(Event(0.0, lpf_callback))

# ========================
#  SIMULATION LOOP
# ========================

# We'll record the LPF output each time it updates
times_lpf = Float64[]
vals_lpf  = Float64[]

current_time = 0.0
while true
    event = pop_next_event()
    event === nothing && break
    current_time = event.time
    event.callback(current_time)

    # After the lpf_callback, the output changes
    if event.callback === lpf_callback
        push!(times_lpf, lpf_block.output.time)
        push!(vals_lpf,  lpf_block.output.value)
    end

    if current_time > T_max
        break
    end
end

println("Simulation complete. Collected $(length(vals_lpf)) LPF samples.")

# ========================
#  TIME-DOMAIN PLOT
# ========================
plot(times_lpf, vals_lpf, label="LPF Impulse Response",
     xlabel="Time (s)", ylabel="Amplitude", lw=2,
     title="Impulse Response in Time Domain")

# ========================
#  FFT -> BODE PLOT
# ========================

# 1) Optional: Window or zero-pad
function hann_window!(x::Vector)
    N = length(x)
    for i in 1:N
        x[i] *= 0.5 - 0.5*cos(2π*(i-1)/(N-1))
    end
end

# --- After the simulation loop, suppose you have recorded the LPF impulse response in vals_lpf ---
N_raw = length(vals_lpf)
pad_factor = 2
N_padded = pad_factor * N_raw

# Normalize the impulse response:
normalized_response = vals_lpf ./ dt  # Now the area is approximately 1

# Optionally, apply a Hann window to reduce spectral leakage
function hann_window!(x::Vector)
    N = length(x)
    for i in 1:N
        x[i] *= 0.5 - 0.5*cos(2π*(i-1)/(N-1))
    end
end
hann_window!(normalized_response)

# Zero-pad the normalized, windowed response:
resp_padded = vcat(normalized_response, zeros(N_padded - N_raw))

# Compute FFT
using FFTW  # FFTW is part of the Julia standard library
H = fft(resp_padded)
fs = 1/dt
freqs = fs .* (0:(N_padded-1)) ./ N_padded

# Compute magnitude (dB) and phase (radians) from FFT result
mag = abs.(H)
mag_db = 20 .* log10.(mag .+ 1e-12)
phase_radians = angle.(H)

# Unwrap phase
function unwrap_phase(ph::Vector{Float64})
    out = copy(ph)
    for i in 2:length(out)
        diff = out[i] - out[i-1]
        if diff > π
            out[i:end] .-= 2π
        elseif diff < -π
            out[i:end] .+= 2π
        end
    end
    return out
end

phase_unwrapped = unwrap_phase(phase_radians)
phase_deg = phase_unwrapped .* (180/π)

# Plotting only positive frequencies (and skipping the DC bin to avoid log10(0))
N_half = div(N_padded, 2)
idx_range = 2:N_half  # Skip DC

f_plot = freqs[idx_range]
mag_plot = mag_db[idx_range]
phase_plot = phase_deg[idx_range]

using Plots
plt_bode_mag = plot(f_plot, mag_plot,
    xscale=:log10, 
    xlabel="Frequency (Hz)", 
    ylabel="Magnitude (dB)", 
    title="Bode Plot: Magnitude", 
    lw=2)
plt_bode_phase = plot(f_plot, phase_plot,
    xscale=:log10, 
    xlabel="Frequency (Hz)", 
    ylabel="Phase (deg)", 
    title="Bode Plot: Phase", 
    lw=2)
plot(plt_bode_mag, plt_bode_phase, layout=(2,1), legend=false)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


using Plots

# ===========================================
#          FD-MODE BLOCK DEFINITIONS
# ===========================================

# Frequency-domain simulation mode flag.
const fd_mode = true  # When true, we use f-domain simulation.

# Define a SineSourceBlock.
# For an f-domain simulation, we can simply treat the source as having a constant gain.
mutable struct SineSourceBlock
    frequency::Float64  # (Not used in the tf, but could be if you want a phase shift, etc.)
    amplitude::Float64
end

# Define a LowPassFilterBlock.
# Its s-domain transfer function is:  H(s) = 1/(1+s*tau)
mutable struct LowPassFilterBlock
    R::Float64
    C::Float64
    tau::Float64
end

# Define a ResistorLoadBlock.
# In our case, if we're measuring voltage, this block passes the voltage through.
mutable struct ResistorLoadBlock
    R::Float64  # For current calculation, I(s)=V(s)/R.
end

# ===========================================
#        TRANSFER FUNCTION METHODS
# ===========================================

# For the source, we assume a voltage source with amplitude A.
function tf(block::SineSourceBlock, s)
    # Here we simply return the amplitude (i.e. a constant gain).
    # (You might also choose to include an s-domain phase shift if desired.)
    return block.amplitude
end

# For the LPF, the transfer function is:
#   H(s) = 1/(1+s*tau)
function tf(block::LowPassFilterBlock, s)
    return 1 / (1 + s * block.tau)
end

# For the resistor load block, if we assume we're measuring voltage,
# the transfer function is unity. (To compute current, you could divide by R.)
function tf(block::ResistorLoadBlock, s)
    return 1
end

# ===========================================
#       CREATE FD BLOCK INSTANCES
# ===========================================

# For our example, let’s use:
# - A source with amplitude 1 (unity gain)
# - An RC low-pass filter with R=1000Ω and C=1e-6 F, so tau = 1e-3 s.
# - A resistor load block (which just passes the voltage through).

sine_block_fd = SineSourceBlock(50.0, 1.0)  # Frequency is not used here.
lpf_block_fd = LowPassFilterBlock(1000.0, 1e-6, 1000.0 * 1e-6)
resistor_block_fd = ResistorLoadBlock(1000.0)

# ===========================================
#      CALCULATE OVERALL TRANSFER FUNCTION
# ===========================================

# If the blocks are connected in series, the overall transfer function is
# the product of the individual transfer functions.
function overall_tf(s)
    return tf(sine_block_fd, s) * tf(lpf_block_fd, s) * tf(resistor_block_fd, s)
end

# ===========================================
#         BODE PLOT CALCULATION
# ===========================================

# Define a frequency range (in Hz) over which to evaluate the transfer function.
f_range = 10 .^ range(0, stop=5, length=500)  # e.g. 1 Hz to 100 kHz
w_range = 2π .* f_range  # Angular frequencies

# Evaluate the overall transfer function at s = jω.
H = [ overall_tf(im * w) for w in w_range ]

# Calculate magnitude in dB and phase in degrees.
mag = 20 .* log10.(abs.(H))
phase = angle.(H) .* (180 / π)

# For a first-order low-pass filter, we expect:
# - Magnitude: flat (0 dB) in the passband, rolling off at high frequencies.
# - Phase: approaching -90° at high frequencies.
# (Note: our source and load are unity, so the overall TF is H(s)=1/(1+s*tau).)

# Plot the Bode magnitude and phase.
plt_bode_mag = plot(f_range, mag,
    xscale = :log10, xlabel = "Frequency (Hz)",
    ylabel = "Magnitude (dB)", title = "Bode Plot: Magnitude", lw = 2)
plt_bode_phase = plot(f_range, phase,
    xscale = :log10, xlabel = "Frequency (Hz)",
    ylabel = "Phase (deg)", title = "Bode Plot: Phase", lw = 2)
plot(plt_bode_mag, plt_bode_phase, layout = (2, 1), legend = false)


## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
# SERDES

using Plots
using DSP
using Statistics
using FFTW
using Random

"""
PAM-4, or 4-level Pulse Amplitude Modulation, is a digital modulation technique
where two bits are encoded into each symbol using four discrete amplitude levels.
This simulator models transmitting a PAM-4 signal through a channel with frequency-dependent
losses and analyzes the received signal quality.
"""
module PAM4Simulator

using Random
using Statistics
using FFTW
using Plots

import Plots: plot, plot!, annotate!, text

export generate_pam4_signal, apply_channel, receive_signal, analyze_signal, plot_eye_diagram

# PAM-4 levels (standardized voltage levels, normalized)
const PAM4_LEVELS = [-3.0, -1.0, 1.0, 3.0]

"""
Generate a PAM-4 signal with optional jitter
"""
function generate_pam4_signal(num_symbols::Int, 
                             samples_per_symbol::Int;
                             jitter_std::Float64=0.05,
                             amplitude_noise_std::Float64=0.1,
                             random_seed::Int=42)
    
    Random.seed!(random_seed)
    
    # Generate random 2-bit symbols (0-3)
    symbols = rand(0:3, num_symbols)
    
    # Map symbols to PAM-4 voltage levels
    symbol_voltages = PAM4_LEVELS[symbols .+ 1]
    
    # Initialize the full waveform
    signal_length = num_symbols * samples_per_symbol
    signal = zeros(Float64, signal_length)
    
    # Apply pulse shaping and jitter
    for i in 1:num_symbols
        # Apply timing jitter
        jitter = round(Int, randn() * jitter_std * samples_per_symbol)
        jitter = clamp(jitter, -samples_per_symbol÷4, samples_per_symbol÷4)
        
        # Calculate positions with jitter
        start_idx = (i-1) * samples_per_symbol + 1 + jitter
        if start_idx < 1
            start_idx = 1
        end
        
        end_idx = min(i * samples_per_symbol + jitter, signal_length)
        if start_idx > end_idx
            continue
        end
        
        # Add the symbol value plus some amplitude noise
        amplitude_noise = randn() * amplitude_noise_std
        signal[start_idx:end_idx] .= symbol_voltages[i] + amplitude_noise
    end
    
    return Dict(
        "symbols" => symbols,
        "symbol_voltages" => symbol_voltages,
        "waveform" => signal,
        "samples_per_symbol" => samples_per_symbol
    )
end

"""
Apply a channel with frequency-dependent losses to the signal
"""
function apply_channel(signal_dict::Dict, 
                      sample_rate::Float64; 
                      cutoff_freq::Float64=2.0e9,
                      rolloff_db_per_decade::Float64=20.0)
    
    signal = signal_dict["waveform"]
    
    # FFT of the signal
    signal_fft = fft(signal)
    n = length(signal)
    
    # Frequency vector
    freqs = FFTW.fftfreq(n, sample_rate)
    
    # Create channel frequency response (low-pass filter with high-frequency losses)
    channel_response = ones(ComplexF64, n)
    
    for i in 1:n
        f = abs(freqs[i])
        if f > cutoff_freq
            # Apply roll-off in dB (convert to linear scale)
            decades = log10(f / cutoff_freq)
            attenuation_db = decades * rolloff_db_per_decade
            attenuation_linear = 10.0^(-attenuation_db / 20.0)
            channel_response[i] *= attenuation_linear
        end
    end
    
    # Apply channel response
    output_fft = signal_fft .* channel_response
    
    # Convert back to time domain
    output_signal = real(ifft(output_fft))
    
    # Add some additional noise to simulate a realistic channel
    output_signal += randn(length(output_signal)) * 0.05
    
    return Dict(
        "input" => signal_dict,
        "output" => output_signal,
        "channel_response" => channel_response,
        "frequencies" => freqs
    )
end

"""
Receive and sample the signal at optimal sampling points
"""
function receive_signal(channel_output::Dict, samples_per_symbol::Int)
    received_waveform = channel_output["output"]
    original_symbols = channel_output["input"]["symbols"]
    input_waveform = channel_output["input"]["waveform"]
    
    num_symbols = length(original_symbols)
    received_samples = zeros(Float64, num_symbols)
    
    # Sample at the center of each symbol period (assuming perfect clock recovery)
    for i in 1:num_symbols
        sample_idx = (i-1) * samples_per_symbol + samples_per_symbol ÷ 2
        if sample_idx <= length(received_waveform)
            received_samples[i] = received_waveform[sample_idx]
        end
    end
    
    # Detect symbols based on decision thresholds
    decision_thresholds = [-2.0, 0.0, 2.0]  # Thresholds between PAM4 levels
    detected_symbols = zeros(Int, num_symbols)
    
    for i in 1:num_symbols
        sample = received_samples[i]
        
        if sample < decision_thresholds[1]
            detected_symbols[i] = 0
        elseif sample < decision_thresholds[2]
            detected_symbols[i] = 1
        elseif sample < decision_thresholds[3]
            detected_symbols[i] = 2
        else
            detected_symbols[i] = 3
        end
    end
    
    return Dict(
        "original_symbols" => original_symbols,
        "received_samples" => received_samples,
        "detected_symbols" => detected_symbols,
        "input_waveform" => input_waveform,
        "output_waveform" => received_waveform,
        "samples_per_symbol" => samples_per_symbol
    )
end

"""
Analyze signal quality metrics
"""
function analyze_signal(received_signal::Dict)
    original_symbols = received_signal["original_symbols"]
    detected_symbols = received_signal["detected_symbols"]
    received_samples = received_signal["received_samples"]
    
    # Calculate BER
    errors = sum(original_symbols .!= detected_symbols)
    ber = errors / length(original_symbols)
    
    # Estimate SNR
    # Group samples by original symbol level
    level_samples = Dict(level => Float64[] for level in 0:3)
    
    for i in 1:length(original_symbols)
        push!(level_samples[original_symbols[i]], received_samples[i])
    end
    
    # Calculate signal power (variance between level means)
    level_means = [mean(get(level_samples, level, [0.0])) for level in 0:3 if !isempty(get(level_samples, level, [0.0]))]
    
    if length(level_means) > 1
        signal_power = var(level_means)
        
        # Calculate noise power (average variance within each level)
        noise_power = mean([var(samples) for (level, samples) in level_samples if length(samples) > 1])
        
        # SNR in dB
        snr_db = 10 * log10(signal_power / max(noise_power, 1e-10))
    else
        snr_db = 0.0  # Not enough different levels detected
    end
    
    # Calculate eye height and width
    eye_heights = []
    
    # Calculate eye height between each adjacent level
    level_data = [(level, get(level_samples, level, Float64[])) for level in 0:3]
    level_data = filter(x -> !isempty(x[2]), level_data)
    
    for i in 1:(length(level_data)-1)
        lower_level = level_data[i]
        upper_level = level_data[i+1]
        
        if !isempty(lower_level[2]) && !isempty(upper_level[2])
            lower_max = maximum(lower_level[2])
            upper_min = minimum(upper_level[2])
            
            eye_height = upper_min - lower_max
            push!(eye_heights, eye_height)
        end
    end
    
    avg_eye_height = isempty(eye_heights) ? 0.0 : mean(eye_heights)
    
    return Dict(
        "BER" => ber,
        "SNR_dB" => snr_db,
        "eye_height" => avg_eye_height,
        "errors" => errors,
        "total_symbols" => length(original_symbols)
    )
end

"""
Plot eye diagram from the received waveform
"""
function plot_eye_diagram(received_signal::Dict; eye_spans::Int=2)
    waveform = received_signal["output_waveform"]
    samples_per_symbol = received_signal["samples_per_symbol"]
    
    # Ensure we have enough data for the eye diagram
    if length(waveform) < eye_spans * samples_per_symbol
        error("Not enough data for eye diagram")
    end
    
    # Extract complete symbols for eye diagram
    num_complete_symbols = length(waveform) ÷ samples_per_symbol
    
    # Initialize plot
    p = Plots.plot(title="PAM-4 Eye Diagram", 
             xlabel="Time (UI)", 
             ylabel="Amplitude",
             legend=false,
             xlims=(0, eye_spans),
             grid=true)
    
    # Plot each eye trace
    for i in 1:(num_complete_symbols - eye_spans + 1)
        start_idx = (i-1) * samples_per_symbol + 1
        end_idx = start_idx + eye_spans * samples_per_symbol - 1
        
        if end_idx <= length(waveform)
            trace = waveform[start_idx:end_idx]
            x_values = range(0, eye_spans, length=length(trace))
            
            Plots.plot!(p, x_values, trace, color=:blue, alpha=0.2)
        end
    end
    
    # If we have analysis metrics, annotate the eye diagram
    if haskey(received_signal, "analysis") && haskey(received_signal["analysis"], "eye_height")
        eye_height = received_signal["analysis"]["eye_height"]
        snr_db = received_signal["analysis"]["SNR_dB"]
        ber = received_signal["analysis"]["BER"]
        
        # Add annotations
        Plots.annotate!(p, [(eye_spans/2, 4, Plots.text("Eye Height: $(round(eye_height, digits=2))", 8, :red)),
                      (eye_spans/2, 3.5, Plots.text("SNR: $(round(snr_db, digits=2)) dB", 8, :red)),
                      (eye_spans/2, 3.0, Plots.text("BER: $(ber)", 8, :red))])
        
        # Draw eye height visualization
        for level in 0:2
            # Approximate the eye center vertical position for level boundaries
            center_y = -3 + level * 2
            
            # Draw eye height indicators
            Plots.plot!(p, [eye_spans/2-0.2, eye_spans/2+0.2], [center_y + eye_height/2, center_y + eye_height/2], 
                  color=:red, linewidth=2)
            Plots.plot!(p, [eye_spans/2-0.2, eye_spans/2+0.2], [center_y - eye_height/2, center_y - eye_height/2], 
                  color=:red, linewidth=2)
            Plots.plot!(p, [eye_spans/2, eye_spans/2], [center_y - eye_height/2, center_y + eye_height/2], 
                  color=:red, linewidth=2, linestyle=:dash)
        end
    end
    
    return p
end

end # module

# Main script to run the simulator

using .PAM4Simulator
using Random
using Plots

function run_simulation(;
    num_symbols=1000,
    samples_per_symbol=32,
    sample_rate=32e9, # 32 GSa/s
    jitter_std=0.1,
    amplitude_noise_std=0.05,
    channel_cutoff=3e9, # 3 GHz cutoff
    rolloff_db=20.0 # 20 dB/decade rolloff
)
    # 1. Generate PAM-4 signal
    tx_signal = generate_pam4_signal(num_symbols, samples_per_symbol, 
                                     jitter_std=jitter_std,
                                     amplitude_noise_std=amplitude_noise_std)
    
    # 2. Apply channel effects
    channel_output = apply_channel(tx_signal, sample_rate, 
                                  cutoff_freq=channel_cutoff,
                                  rolloff_db_per_decade=rolloff_db)
    
    # 3. Receive and detect symbols
    rx_signal = receive_signal(channel_output, samples_per_symbol)
    
    # 4. Analyze signal quality
    analysis = analyze_signal(rx_signal)
    rx_signal["analysis"] = analysis
    
    # Display results
    println("=== PAM-4 Simulation Results ===")
    println("SNR: $(round(analysis["SNR_dB"], digits=2)) dB")
    println("BER: $(analysis["BER"]) ($(analysis["errors"]) errors in $(analysis["total_symbols"]) symbols)")
    println("Average Eye Height: $(round(analysis["eye_height"], digits=2))")
    
    # 5. Visualize results
    # Original and received waveform
    p1 = Plots.plot(tx_signal["waveform"][1:min(500, end)], 
              label="Transmitted", 
              title="PAM-4 Signal Comparison (first 500 samples)",
              ylabel="Amplitude")
    Plots.plot!(p1, channel_output["output"][1:min(500, end)], 
          label="Received", alpha=0.7)
    
    # Channel frequency response
    freqs = abs.(channel_output["frequencies"][1:div(end,2)])
    response_db = 20 .* log10.(abs.(channel_output["channel_response"][1:div(end,2)]))
    p2 = Plots.plot(freqs[2:end] ./ 1e9, response_db[2:end], 
              title="Channel Frequency Response",
              xlabel="Frequency (GHz)", 
              ylabel="Magnitude (dB)",
              xscale=:log10)
    
    # Symbol constellation
    p3 = Plots.scatter(1:length(rx_signal["received_samples"]), 
                 rx_signal["received_samples"],
                 marker=:circle, 
                 markersize=2,
                 markerstrokewidth=0,
                 alpha=0.5,
                 label="Received Samples",
                 title="Symbol Constellation",
                 xlabel="Symbol Index", 
                 ylabel="Amplitude")
    
    # Highlight errors
    error_indices = findall(rx_signal["original_symbols"] .!= rx_signal["detected_symbols"])
    if !isempty(error_indices)
        Plots.scatter!(p3, error_indices, 
                rx_signal["received_samples"][error_indices],
                marker=:x, 
                color=:red, 
                label="Errors")
    end
    
    # Plot ideal PAM-4 levels
    for (i, level) in enumerate([-3.0, -1.0, 1.0, 3.0])
        Plots.hline!(p3, [level], linestyle=:dash, color=:black, label=(i==1 ? "PAM-4 Levels" : ""))
    end
    
    # Eye diagram
    p4 = plot_eye_diagram(rx_signal, eye_spans=2)
    
    # Combine plots
    p_combined = Plots.plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700))
    display(p_combined)
    
    return Dict(
        "tx_signal" => tx_signal,
        "channel_output" => channel_output, 
        "rx_signal" => rx_signal,
        "analysis" => analysis,
        "plots" => (p1, p2, p3, p4, p_combined)
    )
end

# Run with default parameters
results = run_simulation()

# Example of running with different channel conditions
function compare_channel_conditions()
    println("\n=== Comparing Different Channel Conditions ===\n")
    
    # Good channel (less loss)
    println("Good Channel (5 GHz cutoff, 15 dB/decade rolloff):")
    good_results = run_simulation(channel_cutoff=5e9, rolloff_db=15.0, jitter_std=0.05)
    
    # Medium channel (default)
    println("\nMedium Channel (3 GHz cutoff, 20 dB/decade rolloff):")
    medium_results = run_simulation(channel_cutoff=3e9, rolloff_db=20.0, jitter_std=0.1)
    
    # Bad channel (more loss)
    println("\nBad Channel (2 GHz cutoff, 25 dB/decade rolloff):")
    bad_results = run_simulation(channel_cutoff=1.4e9, rolloff_db=25.0, jitter_std=0.45)
    
    # Compare SNR and BER
    results = [
        ("Good", good_results["analysis"]["SNR_dB"], good_results["analysis"]["BER"]),
        ("Medium", medium_results["analysis"]["SNR_dB"], medium_results["analysis"]["BER"]),
        ("Bad", bad_results["analysis"]["SNR_dB"], bad_results["analysis"]["BER"])
    ]
    
    println("\nChannel Quality Comparison:")
    println("Channel\tSNR (dB)\tBER")
    println("---------------------------------")
    for (label, snr, ber) in results
        println("$(label)\t$(round(snr, digits=2))\t\t$(ber)")
    end
    
    return Dict(
        "good" => good_results,
        "medium" => medium_results,
        "bad" => bad_results
    )
end

# Uncomment to run channel comparison
comparison = compare_channel_conditions()

#%%

##############################
# RLC Simulator in Julia
##############################

using DifferentialEquations
using Plots
using ControlSystems

# Set common plot backend (optional)
gr()

##############################
# Part 1: DC Simulation
##############################

# RC Circuit DC Simulation (Step 1V)
R_rc = 1.0         # Resistance in ohms
C = 1.0            # Capacitance in farads

# ODE: dVc/dt = (1 - Vc)/(R*C)
function rc!(du, u, p, t)
    du[1] = (1 - u[1])/(R_rc * C)
end
u0_rc = [0.0]      # Initial capacitor voltage is 0 V
tspan = (0.0, 5.0)
prob_rc = ODEProblem(rc!, u0_rc, tspan)
sol_rc = solve(prob_rc, Tsit5())

# Extract results
t_rc = sol_rc.t
Vc = [u[1] for u in sol_rc.u]
# Resistor current: I = (Vin - Vc)/R (Vin=1V)
I_rc = [(1 - v)/R_rc for v in Vc]

# Plot RC results
p1 = plot(t_rc, Vc, label="Capacitor Voltage (V)", xlabel="Time (s)", ylabel="Voltage (V)", title="RC DC Transient")
plot!(p1, t_rc, I_rc, label="Resistor Current (A)")

# RL Circuit DC Simulation (Step 1V)
R_rl = 1.0         # Resistance in ohms
L = 1.0            # Inductance in henries

# ODE: dI/dt = (1 - R*I)/L
function rl!(du, u, p, t)
    du[1] = (1 - R_rl*u[1])/L
end
u0_rl = [0.0]      # Initial current is 0 A
prob_rl = ODEProblem(rl!, u0_rl, tspan)
sol_rl = solve(prob_rl, Tsit5())

# Extract RL results
t_rl = sol_rl.t
I_rl = [u[1] for u in sol_rl.u]

# Estimate inductor voltage: V_L = L * dI/dt (numerical derivative)
V_L = [L * (I_rl[i+1] - I_rl[i])/(t_rl[i+1]-t_rl[i]) for i in 1:length(t_rl)-1]
t_VL = t_rl[1:end-1]

# Plot RL results
p2 = plot(t_rl, I_rl, label="Current (A)", xlabel="Time (s)", ylabel="Current (A)", title="RL DC Transient")
plot!(p2, t_VL, V_L, label="Inductor Voltage (V)")

##############################
# Part 2: RC Low/High Pass Filters (Bode Plots)
##############################

R_filter = 1.0
C_filter = 1.0

# RC Low Pass: H(s) = 1/(1+RC s)
H_lp = tf(1, [R_filter*C_filter, 1])

# RC High Pass: H(s) = RC s/(1+RC s)
H_hp = tf([R_filter*C_filter, 0], [R_filter*C_filter, 1])

p3 = bodeplot(H_lp, title="RC Low Pass Filter Bode Plot")
p4 = bodeplot(H_hp, title="RC High Pass Filter Bode Plot")

##############################
# Part 3: LC Filters (Bode Plots)
##############################

# For LC filters we introduce a load resistor to provide damping.
L_lc = 1.0
C_lc = 1.0
R_load = 1.0   # small resistor to avoid singular behavior

# LC Bandpass Filter:
# Circuit: series LC with load R. Transfer function from input to load:
# H(s) = R_load/(R_load + sL + 1/(sC))
# Multiply numerator and denominator by sC:
# H(s) = (R_load*s*C)/(s^2*L*C + s*R_load*C + 1)
num_bp = [R_load * C_lc, 0.0]        # Coefficients for R_load * s * C
den_bp = [L_lc * C_lc, R_load * C_lc, 1.0]
H_bp_lc = tf(num_bp, den_bp)

# LC Bandstop Filter:
# Use the standard second order bandstop form:
ω0 = 1/sqrt(L_lc * C_lc)
# Define an arbitrary Q factor (here chosen based on R_load)
Q_lc = R_load * sqrt(C_lc/L_lc)
H_bs_lc = tf([1.0, 0.0, ω0^2], [1.0, ω0/Q_lc, ω0^2])

p5 = bodeplot(H_bp_lc, title="LC Bandpass Filter Bode Plot")
p6 = bodeplot(H_bs_lc, title="LC Bandstop Filter Bode Plot")

##############################
# Part 4: RLC Filters (Bode Plots and Q Factor)
##############################

R_rlc = 1.0
L_rlc = 1.0
C_rlc = 1.0
ω0_rlc = 1/sqrt(L_rlc * C_rlc)
# For a series RLC, Q factor is defined as:
Q_rlc = ω0_rlc * L_rlc / R_rlc

# RLC Bandpass Filter (standard form):
# H(s) = (ω0/Q * s)/(s^2 + (ω0/Q)*s + ω0^2)
H_bp_rlc = tf([ω0_rlc/Q_rlc, 0.0], [1.0, ω0_rlc/Q_rlc, ω0_rlc^2])

# RLC Bandstop Filter (standard form):
# H(s) = (s^2+ω0^2)/(s^2 + (ω0/Q)*s + ω0^2)
H_bs_rlc = tf([1.0, 0.0, ω0_rlc^2], [1.0, ω0_rlc/Q_rlc, ω0_rlc^2])

p7 = bodeplot(H_bp_rlc, title="RLC Bandpass Filter Bode Plot")
p8 = bodeplot(H_bs_rlc, title="RLC Bandstop Filter Bode Plot")

println("RLC Bandpass Filter Q factor: ", Q_rlc)

##############################
# Display all plots
##############################

# You can display the time domain simulations:
display(p1)  # RC transient
display(p2)  # RL transient

# And the filter Bode plots:
display(p3)  # RC Low Pass
display(p4)  # RC High Pass
display(p5)  # LC Bandpass
display(p6)  # LC Bandstop
display(p7)  # RLC Bandpass
display(p8)  # RLC Bandstop


