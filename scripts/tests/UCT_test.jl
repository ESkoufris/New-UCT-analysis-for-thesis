# --- Setup ---
include("../../MCTSUtils.jl")

using .MCTSUtils
using Plots
using Random

mkpath("figures/final/UCT")

# --- Fixed params ---
S = 10
γ = 0.9
mcts_iterations = 100_000
n_simulations   = 100
c_param         = 1
verbose_flag    = false
logscale        = "none"
deterministic_rewards_flag = false

A_list = 2:3
H_list = 2:3

plots = []

for (iA, A) in enumerate(A_list)
    for (iH, H) in enumerate(H_list)
        Random.seed!(0xC0FFEE + 17*A + H)
        means, mdp = random_MDP(S, A; γ=γ, is_deterministic=true, horizon=H, seed=rand(1:10^9), 
                                deterministic_rewards = deterministic_rewards_flag)

        avg_returns, T_samples = run_MCTS(
            mcts_iterations = mcts_iterations,
            n_simulations   = n_simulations,
            mdp             = mdp,
            H               = H,
            verbose         = verbose_flag,
            c_param         = c_param
        )

        p = compute_regret_at_root(mdp, avg_returns, T_samples; divide_by_log = false, logscale = logscale)
        plot!(p;
            legend    = false,
            linewidth = 3,
            xlabel    = "",
            ylabel    = "",
            title     = "")  # remove small subplot titles

        # Add annotations for A and H in large font
        annotate!(p, (0.05, 0.9, text("A=$A", 10, :left, :black)))
        annotate!(p, (0.75, 0.1, text("H=$H", 10, :right, :black)))

        push!(plots, p)
    end
end

layout = @layout [grid(length(A_list), length(H_list))]

bigplot = plot(
    plots...;
    layout = layout,
    size   = (3000, 3000),
    title  = "UCT Regret across A ∈ 5:8 and H ∈ 5:8"
)

savefig(bigplot, "figures/final/UCT/new_fig_with_nonlog.png")


