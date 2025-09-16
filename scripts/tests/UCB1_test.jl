include("../../MCTSUtils.jl")
using .MCTSUtils   # note the dot: means “relative import”
using Plots, Distributions

S = 1
A = 8
gamma = 1
H = 1
n_simulations = 200
iterations = 1000

means, mdp = random_MDP(S, A; γ=gamma, is_deterministic=true, horizon=H, seed = 3)
optimal_arm = argmax(means)[2]

# get and process results
_,visit_counts = run_MCTS(mdp = mdp, mcts_iterations = iterations, n_simulations = n_simulations)

# visit_counts has size (n_simulations, A, iterations)

failure_sums = sum(visit_counts[:, setdiff(1:A, optimal_arm), :], dims = 2)[:,1,:]
println(size(failure_sums))
x_grid = 1:iterations

tail_probs = [mean(failure_sums[:, n] .>= log(x)) for n in 1:iterations]
# println(tail_probs)
# # plot P( ∑ 1{Iⱼ ≠ iꜛ } ≥ log(n))
