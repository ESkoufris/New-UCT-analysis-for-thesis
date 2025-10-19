include("MDP.jl")

function value_iteration(MDP; θ = 1e-8, max_iters = 1000)
    num_states = length(MDP.states)
    num_actions = length(MDP.actions)
    γ = MDP.gamma
    H = MDP.horizon + 1 # equals 1 if horizon is 0 → UCB1 case
    reward_function = get_reward_function(MDP)

    if isinf(H)
        # ----- Infinite horizon: standard fixed point iteration -----
        V = zeros(num_states)
        π = zeros(Int, num_states)
        Q = zeros(num_states, num_actions)

        for iter in 1:max_iters
            Δ = 0.0
            V_new = similar(V)

            for s in 1:num_states
                q_values = zeros(num_actions)
                for a in 1:num_actions
                    for s′ in 1:num_states
                        p = MDP.dynamics[s′, s, a]
                        r = reward_function(s, a)
                        q_values[a] += p * (r + γ * V[s′])
                    end
                end
                Q[s, :] .= q_values
                V_new[s], π[s] = findmax(q_values)
                Δ = max(Δ, abs(V_new[s] - V[s]))
            end

            V = V_new
            if Δ < θ
                break
            end
        end
        return V, π, Q

    else
        # ----- Finite horizon: backward induction -----
        V = zeros(H+1, num_states)   # V[t, s] = value at state s with t steps remaining
        π = zeros(Int, H, num_states)  # policy should be time-dependent
        Q = zeros(H, num_states, num_actions)

        # terminal values
        V[H+1, :] .= 0

        for t in (H):-1:1
            for s in 1:num_states
                q_values = zeros(num_actions)
                for a in 1:num_actions
                    r = reward_function(s, a)
                    q_values[a] += r 
                    for s′ in 1:num_states
                        p = MDP.dynamics[s′, s, a]
                        q_values[a] += p * γ * V[t+1, s′]
                    end
                end
                Q[t, s, :] = q_values
                V[t, s], π[t, s] = findmax(q_values)
            end
        end

        return V[1, :], π, Q
    end
end