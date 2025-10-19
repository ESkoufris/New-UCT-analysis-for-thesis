include("../../MCTSUtils.jl")
using .MCTSUtils
using Plots

# --- Parameters ---
S = 10
A = 5
γ = 1
H = 0   # single horizon

# --- Helper functions ---

"""
Extract the mean rewards for each action at the root state.
Adjust this function if your `means` object has a different structure.
"""
function root_action_means(means, A::Int)
    if means isa AbstractVector
        return means[1:A]
    elseif means isa AbstractMatrix
        return vec(means[1, 1:A])
    elseif means isa Dict
        # assumes keys like (state, action)
        return [means[(1, a)] for a in 1:A if haskey(means, (1, a))]
    else
        error("Unrecognized structure of `means`.")
    end
end

"""
Compute the theoretical UCB1 regret upper bound:
R_n ≤ 8 Σ (log n / Δ_j) + (1 + π²/3) Σ Δ_j
"""
function ucb1_bound(ns::AbstractVector{<:Integer}, deltas::AbstractVector{<:Real})
    positive = findall(>(0), deltas)
    Δ = deltas[positive]
    const_term = (1 + π^2 / 3) * sum(Δ)
    return [8 * sum(log(n) ./ Δ) + const_term for n in ns]
end

# --- Run experiment ---

means, mdp = random_MDP(S, A; γ=γ, is_deterministic=true, horizon=H, seed=3)

average_sim_returns, T_samples = run_MCTS(
    mcts_iterations = 10000,
    n_simulations   = 500,
    mdp             = mdp,
    H               = H,
    verbose         = false,
    c_param         = 1
)

# --- Plot empirical regret ---
compute_regret_at_root(mdp, average_sim_returns, T_samples;
                       logscale = "none", divide_by_log = false)

# # --- Compute and plot theoretical UCB1 upper bound ---
# μ = root_action_means(means, A)
# μ_star = maximum(μ)
# deltas = μ_star .- μ                   # Δ_j = μ* - μ_j
# ns = sort(unique(T_samples))           # evaluation points
# bound_vals = ucb1_bound(ns, deltas)

# plot!(ns, bound_vals;
#       lw = 2, ls = :dash, label = "Theoretical upper bound", legend =:right)

xlabel!("n")
ylabel!("Regret")
# title!("Empirical regret vs theoretical regret bound (UCB1)")
# legend(:topleft)

savefig("figures/final/UCB1/new_empirical_vs_theoretical_regret.png")

