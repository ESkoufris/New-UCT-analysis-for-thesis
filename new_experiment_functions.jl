##########################
# MCTS running functions #
##########################
include("new_MCTS.jl")
include("value_iteration.jl")
using Distributions


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
function compute_regret_at_root(mdp::MDP, all_node_returns, T_samples;
    root_state = 1,
    logscale = "none"   # options: "none", "x", "y", "xy"
)
    n_simulations, A, mcts_iterations = size(all_node_returns)
    S = length(mdp.states)

    optimal_q_values = value_iteration(mdp)[3]
    arms = collect(1:A)
    arm_means = optimal_q_values[1, root_state, :]
    # println(arm_means)
    optimal_mean = maximum(arm_means)

    # average estimated value across runs for each action at each iteration
    mc_estimates = sum(all_node_returns, dims = 1) ./ n_simulations  # (1, A, T)
    mc_estimates = dropdims(mean(all_node_returns, dims = 1), dims = 1)
    # best action estimate at each iteration
    empirical_returns = vec(sum(mc_estimates, dims = 1))  # (T,)

    # now compute n * | E[estimate] - mu* |
    scaled_error = abs.(empirical_returns .- (1:mcts_iterations) .*optimal_mean)

    # determine y-limits to always include optimal_mean
    # ymin = min(minimum(mc_estimates), optimal_mean) - 1
    # ymax = max(maximum(mc_estimates), optimal_mean) + 1
    
    # base plot
    q = plot(1:mcts_iterations, scaled_error,
        xlabel = L"t",
        ylabel = "Regret",
        title  = "Regret over $n_simulations runs",
        # ylim   = (ymin, ymax),
        label  = "Regret"
    )

    # # dashed reference line at the optimal arm's mean reward
    # hline!(q, [optimal_mean], linestyle = :dash, label = "Optimal μ")

    # apply log scaling if requested
    if logscale == "x"
        plot!(q, xscale = :log10)
    elseif logscale == "y"
        plot!(q, yscale = :log10)
    elseif logscale == "xy"
        plot!(q, xscale = :log10, yscale = :log10)
    end

    display(q)
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
