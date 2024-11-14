struct ValueIteration
    k_max  # maximum number of iterations
end

struct MDP
    gamma  # discount factor
    S  # state space
    A  # action space
    T  # transition function
    R  # reward function
end
struct ValueFunctionPolicy
    mdp # problem
    U # utility function
end

function greedy(mdp::MDP, U, s)
    u, alpha = findmax(alpha -> lookahead(mdp, U, s, alpha), mdp.A)
    return (a=alpha, u=u)
end

function lookahead(mdp::MDP, U, s, alpha)
    S, T, R, gamma = mdp.S, mdp.T, mdp.R, mdp.gamma
    return R(s, alpha) + gamma * sum(T(s, alpha, s_next) * U[i] for (i, s_next) in enumerate(S))
end

function backup(mdp::MDP, U, s)
    return maximum(lookahead(mdp, U, s, alpha) for alpha in mdp.A)
end

(policy::ValueFunctionPolicy)(s) = greedy(policy.mdp, policy.U, s).a

function solve(M::ValueIteration, mdp::MDP)
    U = [0.0 for s in mdp.S]
    threshold = 0.001
    for k = 1:M.k_max
        U_new = [backup(mdp, U, s) for s in mdp.S]
        println(k)
        if maximum(abs.(U_new - U)) < threshold
            break
        end
        U = U_new
    end
    return ValueFunctionPolicy(mdp, U)
end
