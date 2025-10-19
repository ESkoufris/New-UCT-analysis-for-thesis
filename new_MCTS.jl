include("project.jl")
include("MDP.jl")

#######################
# Main MCTS functions #
#######################

# ===========================
# MCTS Configuration Struct
# ===========================
mutable struct MCTSConfig
    mdp::MDP
    rollout_policy::Array
    max_depth::Int
    score_function::Function
    c_param::Real
end

mutable struct MCTSResults 
    node_returns # of size (a × MCTS iterations)
    visits_at_root_action_nodes
end

# ===========================
# Node Types
# ===========================
abstract type MCTSNode end

mutable struct StateNode <: MCTSNode 
    """
    Represents a state in the MCTS tree.

    Attributes:
    - state: the environment state.
    - parent: parent ActionNode (or nothing if root).
    - children: Dict mapping action index → ActionNode.
    - depth: depth of the node in the tree.
    """
    state
    parent::Union{MCTSNode, Nothing}
    children::Dict{Int, MCTSNode}
    visits::Int
    depth::Int
end

mutable struct ActionNode <: MCTSNode 
    """
    Represents an action taken from a state in the MCTS tree.

    Attributes:
    - action: the action taken.
    - parent: parent StateNode.
    - children: Dict mapping resulting state → StateNode.
    - visits: number of times action node was visited.
    - value: cumulative value/Q estimate for the action.
    """
    action
    parent::Union{StateNode, Nothing}
    children::Dict{Int, StateNode}
    visits::Int
    value::Float64
end

# ===========================
# Node Constructors
# ===========================
function StateNode(state, parent::Union{MCTSNode, Nothing}=nothing, visits::Int = 0, depth::Int=0)::StateNode
    """
    Construct a StateNode with empty children and optional parent/depth.
    """
    children = Dict{Int, MCTSNode}()
    return StateNode(state, parent, children, visits, depth)
end

function ActionNode(action, parent::Union{StateNode})::ActionNode
    """
    Construct an ActionNode with empty children and zero visits/value.
    """
    children = Dict{Int, StateNode}()
    visits = 0
    value = 0.0
    return ActionNode(action, parent, children, visits, value)
end

# ===========================
# Utility Functions
# ===========================
function is_fully_expanded(state_node::StateNode, config::MCTSConfig)::Bool
    """
    Checks if all possible actions from a state node have been expanded.
    """
    # println("Expanded states:")
    # println("\t Child length: $(length(state_node.children)), Action number: $(length(config.mdp.actions))")
    return length(state_node.children) == length(config.mdp.actions)
end

function best_action(state_node::StateNode, config::MCTSConfig, verbose::Bool = false)
    """
    Selects the best action from a fully expanded state node using the provided score function.
    """
    @assert is_fully_expanded(state_node, config)

    score_function = config.score_function
    actions = sort(collect(keys(state_node.children)))
    childs = state_node.children
    scores = [score_function(childs[a].value, childs[a].visits, state_node.visits, config.c_param) for a in actions]

    if verbose
        print("\tAction scores: ")
        for (a, s) in zip(actions, scores)
            print("$(round(s, digits=2)), ")
        end
        println()  # finish the line
    end
    i = argmax(scores)
    return actions[i]
end 

# function Base.show(io::IO, node::MCTSNode)
#     """
#     Print a node for inspection.
#     """
#     print(io, "d=$(node.depth) n=$(node.visits) v=$(node.value)")
# end

# ===========================
# Sampling Functions
# ===========================
function sample_next_state(dynamics, state, action)
    """
    Sample next state from transition dynamics for a given state and action.
    """
    probs = dynamics[:, state, action]
    dist = Categorical(probs)
    next_state = rand(dist)
    return next_state
end

function sample_reward(MDP::MDP, state, action)
    """
    Sample reward based on state and action using reward dynamics.
    """
    probs = MDP.reward_dynamics[state, action, :]
    dist = Categorical(probs)
    i = rand(dist)
    return MDP.rewards[i]
end

function sample_next_action(rollout_policy, state)
    """
    Sample next action from a state according to a rollout policy.
    """
    probs = rollout_policy[state, :]
    dist = Categorical(probs)
    action = rand(dist)
    return action
end

# ===========================
# Simulation and Step
# ===========================
function step(state, action, config::MCTSConfig)
    """
    Perform a single environment step given state and action.
    Returns: reward, next_state
    """
    reward = sample_reward(config.mdp, state, action)
    next_state = sample_next_state(config.mdp.dynamics, state, action)
    return next_state, reward
end

function simulate(action_node, config::MCTSConfig, depth)
    """
    Rollout simulation from an action node using rollout policy.
    Returns total discounted reward estimate.
    """
    state = action_node.parent.state
    action = action_node.action
    next_state,_ = step(state, action, config)    # Get Sₜ₊₁
    total_reward = 0.0
    discount = 1.0

    for d in 1:depth
        action = sample_next_action(config.rollout_policy, next_state) # Get Aₜ₊₁
        next_state, reward = step(next_state, action, config) # Get (Sₜ₊₂, Rₜ₊₁)
        total_reward += discount * reward
        discount *= config.mdp.gamma
        if d == config.max_depth
            break
        end
        state = next_state
    end
    return total_reward
end

# ===========================
# Expansion Functions
# ===========================
function expand!(state_node::StateNode, config::MCTSConfig)::ActionNode
    """
    Expand a state node by adding a new, previously untried action.
    """
    tried_actions = collect(keys(state_node.children))
    untried_actions = [a for a in config.mdp.actions if a ∉ tried_actions]
    @assert !isempty(untried_actions)

    action = rand(untried_actions)
    state_node.children[action] = ActionNode(action, state_node)
    return state_node.children[action]
end

function expand!(action_node::ActionNode, next_state, depth::Int)::StateNode
    """
    Expand an action node by adding the resulting state node if it does not exist.
    """
    states = collect(keys(action_node.children))
    # println("Actio")
    if next_state ∉ states 
        action_node.children[next_state] = StateNode(next_state, action_node, 0, depth)
    end 
    return action_node.children[next_state]
end

# ===========================
# Backpropagation
# ===========================
function backpropagate(trajectory::Array, rollout_reward::Real, config::MCTSConfig)
    """
    Backpropagate rewards from a rollout through the trajectory of action nodes.
    """
    cumulative_reward = rollout_reward
    for (action_node, reward) in reverse(trajectory)
        cumulative_reward = reward + config.mdp.gamma * cumulative_reward
        action_node.visits += 1
        action_node.parent.visits += 1
        action_node.value += cumulative_reward 
    end
end

# ===========================
# Main MCTS Loop
# ===========================
function MCTS(root::StateNode, config::MCTSConfig, iterations=100; verbose::Bool = false, get_returns::Bool = false)
    """
    Run MCTS from a given root state node.

    Parameters:
    - root: StateNode
    - config: MCTSConfig
    - iterations: number of MCTS iterations
    - verbose: if true, prints detailed info per iteration
    - get_returns: if

    Returns: 
    """
    mdp = config.mdp
    if verbose
        mdp = config.mdp
        println("==================")
        println("MDP parameters:")
        println(mdp)
        println("Config max depth:", config.max_depth)
        println("==================")
    end 

    # a×n element is the cumulative returns of action a after n iterations at the root 
    node_returns = zeros(Float64, (length(mdp.actions), iterations)) 

    # i × n element is T_i(n)
    visit_counts = zeros(Int64, (length(mdp.actions), iterations))
    
    for it in 1:iterations
        state_node = root
        trajectory = [] # store (action_node, reward)
        state = state_node.state
        depth = 0

        if verbose
            println("\n\n-------------------")
            println("ITERATION $it")
            println("-------------------")
            println("Starting at root state S0 = $state")
        end
        
        # Selection phase
        if verbose 
            println("\nSelection phase:")
        end  
        while is_fully_expanded(state_node, config) && depth < config.max_depth + 1
            action = best_action(state_node, config, verbose)
            next_state, reward  = step(state, action, config) # Get Sₜ₊₁,Rₜ 
            action_node = state_node.children[action]   # Get action node corresponding to Aₜ
            push!(trajectory, (action_node, reward))    # Add Aₜ,Rₜ to trajectory 
            state_node = expand!(action_node, next_state, depth + 1)  # Get state node corresponding to Sₜ₊₁
            depth += 1
            state = next_state

            if verbose
                println("\t(A$(depth-1) = $action, R$(depth-1) = $reward, S$depth = $next_state)")
            end
        end
        
        # Expansion phase
        if !is_fully_expanded(state_node, config) && depth < config.max_depth + 1
            if verbose
                println("\nExpansion phase:")
                println("\nS$depth = $state is unexpanded")
            end 
            action_node = expand!(state_node, config)    # Get action node corresponding to Aₜ
            reward = sample_reward(config.mdp, state_node.state, action_node.action)   # Get Rₜ
            push!(trajectory, (action_node, reward))   # Add Aₜ,Rₜ to trajectory 
            # depth += 1
            if verbose
                println("\t Expanded: (A$(depth) = $(action_node.action), R$(depth) = $reward)")
            end
        end

        # Simulation
        rollout_reward = simulate(action_node, config, config.max_depth - depth) # Get Rₜ₊₁ + ... 

        if verbose
            println("\nRollout phase:")
            if depth ≥ config.max_depth
                println("\t$(rollout_reward) rollout return; maximum depth reached")
            else 
                if depth == config.max_depth - 1
                    println("\tRollout return: R$(depth+1) = $rollout_reward")
                else 
                    println("\tRollout return: R$(depth + 1) + ... + R$(config.max_depth)  = $rollout_reward")
                end 
            end 
        end

        # Backpropagation
        backpropagate(trajectory, rollout_reward, config)

        # Collect results 
        actions = sort(collect(keys(root.children)))
        all_cumulative_returns = zeros(length(mdp.actions))
        counts = zeros(length(mdp.actions))
        for (i, a) in enumerate(mdp.actions)
            if haskey(root.children, a)
                counts[i] = root.children[a].visits
                all_cumulative_returns[i] = root.children[a].value
            else
                counts[i] = 0
                all_cumulative_returns[i] = 0.0
            end
        end
        node_returns[:, it] = all_cumulative_returns
        visit_counts[:, it] = counts

        # printing 
        if verbose
            println()
            println("Trajectory:")
            for (d, (node, r)) in enumerate(trajectory)
                println("\tS$(d - 1)=$(node.parent.state), A$(d-1)=$(node.action)")
            end
            # values = [childs[a].value for a in actions]
            println("Cumulative action values at root:")
            for a in actions
                println(a,": $(root.children[a].value)")
            end 
            println("===================")
            println("END OF ITERATION $it")
            println("==================")
        end
    end

    # Pick best action by average value
    values = [
        haskey(root.children, a) ? root.children[a].value / root.children[a].visits : -Inf
        for a in config.mdp.actions
    ]

    if verbose
        println("Final root action values:")
        for (a, v) in zip(config.mdp.actions, values)
            println("Action $a → value: $v")
        end
    end

    return argmax(values), MCTSResults(node_returns, visit_counts)
end