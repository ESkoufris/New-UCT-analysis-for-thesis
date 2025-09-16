include("../../MCTSUtils.jl")

using Printf
using .MCTSUtils   

S = 10
A = 3
H = 5
gamma = 0.9 
means, mdp = random_MDP(S, A; γ=gamma, is_deterministic=true, horizon=H, seed = 23)
average_sim_returns, T_samples = run_MCTS(mcts_iterations = 10, 
                                          n_simulations = 1,
                                          mdp = mdp, H=H, verbose=true,
                                          c_param = 5)

optimal_q_values = value_iteration(mdp)[3]
arm_means = optimal_q_values[1, 1, :]
println("Arm means:", arm_means)

