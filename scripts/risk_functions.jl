include("../MCTSUtils.jl")
using .MCTSUtils   # note the dot: means “relative import”
using Plots

S = 10
A = 2
gamma = 0.9
horizons = 3

for h in horizons
    H = h

    means, mdp = random_MDP(S, A; γ=gamma, is_deterministic=true, horizon=H, seed = 3)
    average_sim_returns, T_samples = run_MCTS(mcts_iterations = 10000, 
                                              n_simulations = 500, mdp = mdp, 
                                              H=H, verbose=false,
                                              c_param = 1)

    # # --- Risk functions ---
    # compute_risk_at_root(mdp, average_sim_returns, T_samples)
    # savefig("figures/presentation/horizons/risk_function_normal_newest_$h.png")

    # compute_risk_at_root(mdp, average_sim_returns, T_samples; logx = true, logy = true)
    # savefig("figures/presentation/horizons/risk_function_loglog_newest_$h.png")

    # --- Bias functions ---
    compute_regret_at_root(mdp, average_sim_returns, T_samples; logscale = "none")
    savefig("figures/final/UCT/regret.png")
    
    # compute_regret_at_root(mdp, average_sim_returns, T_samples; logscale = "x")
    # savefig("figures/newest/regret_semilogx_n
    # ewest_$h.png")
end 