##########################
# MCTS running functions #
##########################
include("MDP.jl")
include("new_MCTS.jl")
include("value_iteration.jl")
using Distributions
using Statistics

using Plots

function run_MCTS(
    ; mdp=nothing,
      rollout_type::Symbol=:random,
      rollout_policy_custom=nothing,
      n_simulations::Int = 200,
      gamma=1,
      max_depth=50,
      c_param=1.2,
      S=6,
      A=3,
      H=10,
      mcts_iterations=10_000,
      root_state::Int=1,
      x_grid::Union{Nothing,AbstractVector}=nothing,
      verbose=false
)
"""
Runs MCTS for a number of iterations.

Arguments:

Returns:
    all_node_returns:: A tensor of size (n_simulations, A, mcts_iterations)
    T_samples:: A tensor of size (n_simulations, A, mcts_iterations)
"""
    # --- MDP setup ---
    if isnothing(mdp)
        means, mdp = random_MDP(S, A; γ=gamma, is_deterministic=true, horizon=H)
    else
        means = nothing  # assume user-supplied MDP doesn't expose means
    end
    S, A = length(mdp.states), length(mdp.actions)

    # rollout policy
    rollout_policy = begin
        if rollout_type == :random
            random_rollout(mdp)
        elseif rollout_type == :optimal
            rp = zeros(S, A)
            for s in 1:S, a in 1:A
                rp[s, a] = (optimal_policy[s] == a)
            end
            rp
        elseif rollout_type == :custom
            isnothing(rollout_policy_custom) && error("Custom rollout policy not provided")
            rollout_policy_custom
        else
            error("Invalid rollout_type: choose :random, :optimal, or :custom")
        end
    end

    config = MCTSConfig(mdp, rollout_policy, H, ucb, c_param)

    @assert 1 ≤ root_state ≤ S "root_state out of bounds"

    arms = collect(1:A)

    # helper: extract root visit counts for action a 
    get_root_counts = (root, a) -> root.children[a].visits

    # --- simulate and collect T_i(n) for all arms ---
    T_samples = zeros(Int, n_simulations, length(arms), mcts_iterations)  # rows: runs, cols: arms
    all_node_returns = zeros(Float64, n_simulations, A, mcts_iterations)

    for r in 1:n_simulations
        root = StateNode(root_state)
        _,MCTSResults = MCTS(root, config, mcts_iterations; verbose=verbose)
        all_node_returns[r,:,:] = MCTSResults.node_returns
        T_samples[r, :, :] = MCTSResults.visits_at_root_action_nodes
    end

    return all_node_returns, T_samples
end

function compute_risk_at_root(mdp::MDP, average_node_returns, T_samples;
    root_state::Int = 1,
    x_grid::Union{Nothing,AbstractVector}=nothing,
    eps::Real = 1e-12,
    save_plot::Bool = false,
    logx::Bool = false,
    logy::Bool = false
)
    # --- basic setup ---
    n_simulations, mcts_iterations = size(average_node_returns)
    S, A = length(mdp.states), length(mdp.actions)
    arms = collect(1:A)

    optimal_q_values = value_iteration(mdp)[3]
    arm_means = optimal_q_values[1, root_state, :]

    # --- x grid (avoid too dense by default) ---
    if x_grid === nothing
        step = max(1, ceil(Int, mcts_iterations ÷ 1000))  # ~1000 points max by default
        x_vals = collect(0:step:mcts_iterations)
    else
        x_vals = collect(x_grid)
    end

    # --- empirical survival functions: rows=x, cols=arm ---
    F_vals = [ mean(T_samples[:, i] .>= x) for x in x_vals, i in eachindex(arms) ]
    # clipped copy for log-safe plotting
    F_vals_clipped = max.(F_vals, eps)

    # --- handle log-x: can't plot x==0 on log axis, so drop x==0 entries ---
    if logx && any(x_vals .== 0)
        @warn "x_vals contains 0; dropping x=0 entries for log-x plotting"
        mask = x_vals .> 0
        x_plot = x_vals[mask]
        F_plot = F_vals_clipped[mask, :]
    else
        x_plot = x_vals
        # if plotting with logy we should use clipped values; otherwise keep original
        F_plot = logy ? F_vals_clipped : F_vals
    end

    # --- axis scaling choices for Plots.jl ---
    xscale = logx ? :log10 : :identity
    yscale = logy ? :log10 : :identity

    # --- sort arms by descending mean for plotting order ---
    sorted_idx = sortperm(arm_means, rev = true)

    title_str = "Survival functions of " * L"T_i(n)" * "\n" *
                "over $n_simulations runs (n = $mcts_iterations)"

    # y-limits: for linear y keep (0,1); for log-y set (eps,1)
    ylims = logy ? (eps, 1.0) : (0.0, 1.0)

    # --- first plot (main survival) ---
    first_idx = sorted_idx[1]
    lbl_first = isnan(arm_means[first_idx]) ?
        "Arm $(arms[first_idx])" :
        "Arm $(arms[first_idx]) (μ=$(round(arm_means[first_idx], digits=3)))"

    p = plot(
        x_plot, F_plot[:, first_idx],
        xlabel = L"x",
        ylabel = L"S(x) = P(T_i(n) \ge x)",
        title  = title_str,
        label  = lbl_first,
        xscale = xscale,
        yscale = yscale,
        ylim = ylims
    )

    for idx in sorted_idx[2:end]
        lbl = isnan(arm_means[idx]) ?
            "Arm $(arms[idx])" :
            "Arm $(arms[idx]) (μ=$(round(arm_means[idx], digits=3)))"
        plot!(p, x_plot, F_plot[:, idx], label = lbl)
    end

    display(p)
    if save_plot
        savefig(p, "survival_plot.png")
    end

    # --- optional secondary plot: log S(x) vs x on a linear y-axis (if user wants actual log(S)) ---
    # If the user specifically wanted to plot log(S(x)) (not just a log scale), do that:
    if logy == :log_of_S  # If you want this feature, call with logy = :log_of_S
        p_logS = plot(x_plot, log.(max.(F_plot, eps))[:, sorted_idx[1]],
                      xlabel = L"x",
                      ylabel = L"\log S(x)",
                      title = title_str,
                      label = lbl_first)
        for idx in sorted_idx[2:end]
            plot!(p_logS, x_plot, log.(max.(F_plot[:, idx], eps)), label = "Arm $(arms[idx])")
        end
        display(p_logS)
        save_plot && savefig(p_logS, "survival_logS_plot.png")
    end

    return x_vals, F_vals, T_samples, arms, arm_means
end


##############
# Bias plots #
##############
function compute_regret_at_root(
    mdp::MDP,
    all_node_returns,
    T_samples;
    root_state = 1,
    logscale = "none",       # options: "none", "x", "y", "xy"
    divide_by_log = false
)
    n_simulations, A, mcts_iterations = size(all_node_returns)
    S = length(mdp.states)

    # --- Compute optimal value ---
    optimal_q_values = value_iteration(mdp)[3]
    arm_means = optimal_q_values[1, root_state, :]
    rsa = get_reward_function(mdp)
    # Used for debugging in UCB1
    # println([rsa(root_state, a) for a in 1:A])
    # println(arm_means)
    optimal_mean = maximum(arm_means)

    # --- Compute empirical estimates ---
    mc_estimates = dropdims(mean(all_node_returns, dims = 1), dims = 1)  # (A, T)
    empirical_returns = vec(sum(mc_estimates, dims = 1))  # (T,)

    # --- Compute regret ---
    scaled_error = abs.(empirical_returns .- (1:mcts_iterations) .* optimal_mean)

    # --- Optionally divide by log(t) ---
    if divide_by_log
        t_vals = collect(1:mcts_iterations)
        log_terms = log.(max.(t_vals, 2))  # avoid log(1)=0
        scaled_error = scaled_error ./ log_terms
    end

    # --- Base plot ---
    y_label = divide_by_log ? L"\text{Regret} / \log t" : "Regret"

    q = plot(
        1:mcts_iterations,
        scaled_error;
        xlabel = L"t",
        ylabel = y_label,
        label = "Empirical",
        legend = true,
        linewidth = 8
    )

    # --- Apply log scales if requested ---
    if logscale == "x"
        plot!(q, xscale = :log10)
    elseif logscale == "y"
        plot!(q, yscale = :log10)
    elseif logscale == "xy"
        plot!(q, xscale = :log10, yscale = :log10)
    end

    return q
end



function compute_expected_pulls(
    mdp::MDP, T_samples;
    root_state = 1,
    logscale = "none"   # options: "none", "x", "y", "xy"
)

    # Number of actions
    A = length(mdp.actions)

    # Average over all simulation runs: E[T_i(n)]
    # average_node_returns should be a dictionary or array where
    # average_node_returns[a][n] = number of times action a pulled by time n
    expected_pulls = Dict(a => mean(average_node_returns[a], dims=1)[:] for a in mdp.actions)

    # Identify optimal arm(s)
    optimal_value = maximum(mean.(values(average_node_returns)))
    optimal_arms = [a for a in mdp.actions if mean(average_node_returns[a]) ≈ optimal_value]

    # Suboptimal arms
    suboptimal_arms = setdiff(mdp.actions, optimal_arms)

    # Set up the plot
    plt = plot(title="Expected Pulls of Suboptimal Arms",
               xlabel="n", ylabel="E[Tᵢ(n)]",
               legend=:topleft)

    # Apply log scaling
    if logscale == "x"
        plot!(plt, xaxis=:log10)
    elseif logscale == "y"
        plot!(plt, yaxis=:log10)
    elseif logscale == "xy"
        plot!(plt, xaxis=:log10, yaxis=:log10)
    end

    # Plot each suboptimal arm
    for a in suboptimal_arms
        plot!(plt, 1:T_samples, expected_pulls[a],
              label="Arm $a", lw=2)
    end

    display(plt)
    return plt
end
### 
# Compute drift 
####



function compute_regret_at_root_with_errors(
    mdp::MDP,
    all_node_returns,
    T_samples;
    root_state = 1,
    logscale = "none",       # "none" | "x" | "y" | "xy"
    divide_by_log = false
)
    n_simulations, A, mcts_iterations = size(all_node_returns)

    # --- Optimal value at the root ---
    optimal_q_values = value_iteration(mdp)[3]            # (A?, S, A)
    arm_means = optimal_q_values[1, root_state, :]        # Q*(s=1, a)
    optimal_mean = maximum(arm_means)

    # --- Per-simulation empirical cumulative returns at the root over time ---
    # Sum across actions to match your original construction
    # Result: (n_simulations, mcts_iterations)
    empirical_returns_sim = dropdims(sum(all_node_returns, dims=2), dims=2)

    # --- Mean and SE across simulations at each time t ---
    mean_empirical = vec(mean(empirical_returns_sim, dims=1))                      # (T,)
    std_empirical  = vec(std(empirical_returns_sim,  dims=1; corrected=true))      # (T,)
    stderr_empirical = std_empirical ./ sqrt(n_simulations)                        # (T,)

    # --- Regret (error) relative to optimal cumulative return t * optimal_mean ---
    t_vals = collect(1:mcts_iterations)
    mean_error = abs.(mean_empirical .- t_vals .* optimal_mean)

    # SE of (X - const) is the same as SE of X; abs() doesn't change SE magnitude.
    stderr_error = copy(stderr_empirical)

    # --- Optional divide by log t ---
    if divide_by_log
        log_terms = log.(max.(t_vals, 2))  # avoid log(1)=0
        mean_error   .= mean_error   ./ log_terms
        stderr_error .= stderr_error ./ log_terms
    end

    # --- Plot: mean ± 1 SE ribbon ---
    y_label = divide_by_log ? L"\text{Regret} / \log t" : "Regret"
    q = plot(
        t_vals, mean_error;
        ribbon = stderr_error,          # ±1 SE shaded band
        xlabel = L"t",
        ylabel = y_label,
        label  = "Mean ± SE",
        linewidth = 3
    )

    # --- Log scales if requested ---
    if logscale == "x"
        plot!(q, xscale = :log10)
    elseif logscale == "y"
        plot!(q, yscale = :log10)
    elseif logscale == "xy"
        plot!(q, xscale = :log10, yscale = :log10)
    end

    return q
end

function compute_regret_at_root_with_variability(
    mdp::MDP, all_node_returns, T_samples;
    root_state=1, logscale="none", divide_by_log=false,
    band=:std  # :std for ±SD, or a tuple like (0.1, 0.9) for quantile band
)
    n_sims, A, T = size(all_node_returns)

    # Optimal cumulative return baseline
    Qstar = value_iteration(mdp)[3]
    optimal_mean = maximum(Qstar[1, root_state, :])
    t_vals = collect(1:T)

    # Per-simulation cumulative return at root over time
    # (adjust this if your notion of "return" is different)
    empirical_returns_sim = dropdims(sum(all_node_returns, dims=2), dims=2)  # (n_sims, mcts_iterations)

    mean_emp = vec(mean(empirical_returns_sim, dims=1))                      # (T, )
    std_emp  = vec(std(empirical_returns_sim, dims=1; corrected=true))       # (mcts_iterations, )
    println(size(std_emp))
    mean_err = abs.(mean_emp .- t_vals .* optimal_mean)

    # Variability band (variance, not SE)
    ribbon_vals = begin
        if band === :std
            std_emp                           # ±1 SD
        elseif band isa Tuple && length(band) == 2
            qlo, qhi = band
            lo = vec(mapslices(x -> quantile(x, qlo), empirical_returns_sim; dims=1))
            hi = vec(mapslices(x -> quantile(x, qhi), empirical_returns_sim; dims=1))
            # convert to error relative to baseline
            lo_err = abs.(lo .- t_vals .* optimal_mean)
            hi_err = abs.(hi .- t_vals .* optimal_mean)
            # Plots.jl ribbon expects symmetric or a 2-row matrix; use 2-row (lower/upper offsets)
            vcat((mean_err .- lo_err)', (hi_err .- mean_err)')  # 2×T
        else
            error("band must be :std or (qlo,qhi) tuple, e.g., (0.25,0.75)")
        end
    end

    if divide_by_log
        logs = log.(max.(t_vals, 2))
        mean_err ./= logs
        if band === :std
            ribbon_vals ./= logs
        else
            ribbon_vals .= ribbon_vals ./ logs'  # broadcast over 2×T
        end
    end

    ylabel = divide_by_log ? L"\text{Error} / \log t" : "Error"
    p = plot(t_vals, mean_err;
        ribbon = ribbon_vals,
        fillalpha = 0.25,
        color = :royalblue,
        linewidth = 4,
        label = (band === :std ? "Mean ± SD" : "Mean with quantile band"),
        xlabel = "Iterations",
        ylabel = ylabel,
    )

    if logscale == "x"
        plot!(p, xscale=:log10)
    elseif logscale == "y"
        plot!(p, yscale=:log10)
    elseif logscale == "xy"
        plot!(p, xscale=:log10, yscale=:log10)
    end

    return p
end