using Printf
using Random
using CSV, DataFrames
using Statistics
using Base: count
using BenchmarkTools

include("value_iteration.jl")
include("q_learning.jl")

function create_small_mdp()
    df = CSV.read("/Users/kennyosele/Documents/Projects/fall24/cs238/repo/project2/data/small.csv", DataFrame)
    gamma = 0.95
    S = collect(1:100)
    A = collect(1:4)

    function T(s::Int, a::Int, sp::Int)
        filtered = filter(row -> row.s == s && row.a == a && row.sp == sp, df)
        total_transitions = filter(row -> row.s == s && row.a == a, df)

        if isempty(filtered)
            return 0.0
        end

        return nrow(filtered) / nrow(total_transitions)
    end

    function R(s::Int, a::Int)
        filtered = filter(row -> row.s == s && row.a == a, df)
        if isempty(filtered)
            return 0.0
        end
        return mean(filtered.r)
    end

    return MDP(gamma, S, A, (s,a,sp) -> T(s,a,sp), (s,a) ->R(s,a))
end

# mdp = create_small_mdp()
# solver = ValueIteration(1)
# policy = solve(solver, mdp)

file_prefix = "/Users/kennyosele/Documents/Projects/fall24/cs238/repo/project2/policies/"
# file_name = file_prefix * "small.policy"
# open(file_name, "w") do f
#     for s in 1:100
#         write(f, @sprintf("%d\n", policy(s)))
#     end
# end



function train_medium_model()
    csv_file_name = "/Users/kennyosele/Documents/Projects/fall24/cs238/repo/project2/data/medium.csv"
    data = CSV.read(csv_file_name, DataFrame)
    states = collect(1:50000)
    actions = collect(1:7)
    gamma = 1.0
    alpha = 0.1

    model = QLearning(states, actions, gamma, zeros(length(states), length(actions)), alpha)

    # train model
    for epoch in 1:1000
        for row in eachrow(data)
            update!(model, row.s, row.a, row.r, row.sp)
        end
    end

    return model
end
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1
model = @btime train_medium_model()
policy = zeros(Int, length(model.S))
for s in model.S
    policy[s] = argmax(model.Q[s,:])
end

file_name = file_prefix * "medium.policy"
open(file_name, "w") do f
    for p in policy
        write(f, @sprintf("%d\n", p))
    end
end


function train_large_model()
    csv_file_name = "/Users/kennyosele/Documents/Projects/fall24/cs238/repo/project2/data/large.csv"
    data = CSV.read(csv_file_name, DataFrame)
    states = collect(1:302020)
    actions = collect(1:9)
    gamma = 1.0
    alpha = 0.1

    model = QLearning(states, actions, gamma, zeros(length(states), length(actions)), alpha)

    # train model
    for epoch in 1:1000
        for row in eachrow(data)
            update!(model, row.s, row.a, row.r, row.sp)
        end
    end

    return model
end

model = train_large_model()
policy = zeros(Int, length(model.S))
for s in model.S
    policy[s] = argmax(model.Q[s,:])
end

file_name = file_prefix * "large.policy"
open(file_name, "w") do f
    for p in policy
        write(f, @sprintf("%d\n", p))
    end
end
