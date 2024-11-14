mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount factor
    Q # action value function
    alpha # learning rate
end

function update!(model::QLearning, s, a, r, s_next)
    Q, gamma, alpha = model.Q, model.gamma, model.alpha
    Q[s,a] += alpha*(r + gamma*maximum(Q[s_next,:]) - Q[s,a])
    return model
end
