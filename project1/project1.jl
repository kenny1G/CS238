#=
To add a pakcage in julia:
enter julia REPL with `julia` command
enter pkg REPL using `]` key
add package with `add <package name>`
=#
import Pkg
# Pkg.add("Graphs")
# Pkg.add("Printf")
# Pkg.add("Plots")
# Pkg.add("GraphRecipes")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("SpecialFunctions")
# Pkg.add("BenchmarkTools")
# Pkg.add("TikzGraphs")
# Pkg.add("TikzPictures")
using Graphs
using Printf
using Plots
using GraphRecipes
using CSV, DataFrames
using SpecialFunctions
using LinearAlgebra
using BenchmarkTools
using TikzGraphs
using TikzPictures
#=
Returns the corresponding linear index for the coordinates x into an array with
dimensions siz.
=#
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

#=
Extracts statistics/counts for a Bayesian network from a dataset (D).
D is n x m matrix. n = num variables, m = num data points

The Bayesian network is defined by variables (vars) and a graph structure (G).


Returns:
- M: An n-length array of matrices, where each matrix contains counts of variable values
     given parent configurations in the Bayesian network
=#
function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    # println("Number of variables: $n")

    r = [vars[i].r for i in 1:n]
    # println("Cardinalities of variables: $r")

    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]
    # println("Number of parent configurations for each variable: $q")

    M = [zeros(q[i], r[i]) for i in 1:n]
    # println("Initialized count matrices M $M")

    # println("Starting to process each data point")
    for (col, o) in enumerate(eachcol(D))
        # Each sample increments n Mijk counts
        for i in 1:n
            k = o[i]
            if k > r[i] || k < 1
                # @warn "Column $col: value $k is out of range for variable $(vars[i].name) with range $(r[i])"
                # println("Full column data: ", o)
                continue
            end
            parents = inneighbors(G, i)
            # q_i is 1 if the variable has no parents
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
                # println("For variable $i, parent configuration index: $j")
            else
                # println("Variable $i has no parents")
            end
            # println("Updating count for variable $(vars[i].name), parent config $j, value $k")
            M[i][j, k] += 1.0
        end
    end
    return M
end

#=
Returns the prior counts for a Bayesian network defined by variables (vars) and a graph structure (G).
=#
function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

function bayesian_score_component(M, alpha)
    p = sum(loggamma.(alpha + M))
    p -= sum(loggamma.(alpha))
    p += sum(loggamma.(sum(alpha, dims=2)))
    p -= sum(loggamma.(sum(alpha, dims=2) + sum(M, dims=2)))
    return p
end

function bayesian_score(vars, G, D::Matrix{Int})
    n = length(vars)
    # Mijk is the number of times the i-th variable takes the value k, given the parents of i take the value j
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in 1:n)
end

# Test the bayesian_score function
struct Variable
    name::Symbol
    r::Int  # number of possible values (cardinality)
end
# Example from video
# q = SimpleDiGraph(3)
# add_edge!(q, 1, 3)
# add_edge!(q, 2, 3)

# vars = [Variable(:A, 2), Variable(:B, 2), Variable(:C, 3)]

# # since julia is 1 indexed, no value in D can be 0
# D = [2 1 1 1 2; 2 2 2 1 1; 2 3 1 2 3]
# score = bayesian_score(vars, q, D)
# println("Example Video Score: ", score)


# # Create a simple test dataset
function test_bayesian_score()
    g = SimpleDiGraph(6)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 4)
    add_edge!(g, 3, 4)
    add_edge!(g, 5, 6)
    add_edge!(g, 5, 4)
    # Read the CSV file

    # Read the CSV file
    df = CSV.read("example/example.csv", DataFrame)

    # Create variables from the dataframe
    vars = [Variable(Symbol(name), maximum(df[!, name])) for name in names(df)]

    # Create the data matrix D
    D::Matrix{Int} = Matrix{Int}(df)' # Transpose to get variables as rows

    println("Variables created:")
    for v in vars
        println(v)
    end

    # println("\nData matrix D:")
    # println(D)
    score = bayesian_score(vars, g, D)
    println("Example ", score)
end

#=
Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
=#
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

struct K2Search
    ordering::Vector{Int} # variable ordering
end

#=
Performs K2 search to find the optimal Bayesian network structure.
=#
function fit(method::K2Search, vars, D)
    # Initialize the graph with no edges
    G = SimpleDiGraph(length(vars))

    # Iterate over the variables according to the provided ordering
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)

        # Greedily add parents to the current node
        while true
            y_best, j_best = -Inf, 0

            # Try adding each possible parent
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    # Temporarily add an edge and compute the new score
                    add_edge!(G, j, i)
                    y′ = bayesian_score(vars, G, D)

                    # Keep track of the best parent
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end

                    # Remove the temporary edge
                    rem_edge!(G, j, i)
                end
            end

            # If adding a parent improves the score, keep it
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                # If no improvement, move to the next variable
                break
            end
        end
    end

    # Return the final graph structure
    return G
end
function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    # 1. Parse CSV file
    # data = CSV.read(infile, DataFrame)
    df = CSV.read(infile, DataFrame)

    # Create variables from the dataframe
    vars = [Variable(Symbol(name), maximum(df[!, name])) for name in names(df)]

    # Create the data matrix D
    D::Matrix{Int} = Matrix{Int}(df)' # Transpose to get variables as rows

    # # 2. Create the Bayesian net

    # println("Variables created:")
    # for v in vars
    #     println(v)
    # end

    # println("\nData matrix D:")
    # println(D)
    ordering = collect(1:length(vars))
    G = fit(K2Search(ordering), vars, D)

    # # 3. Write the gph
    # idx2names = Dict(1 => "Node1", 2 => "Node2")
    idx2names = Dict(i => String(vars[i].name) for i in eachindex(vars))
    write_gph(G, idx2names, "$outfile.gph")
    println("Score of optimal network: ", bayesian_score(vars, G, D))
    # small: 33.960
    # medium: 1.496s
    # large: 204s


    # g = smallgraph(:chvatal)
    # node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


    node_names = [String(v.name)[1:min(4, length(String(v.name)))] for v in vars]
    t = TikzGraphs.plot(G, node_names)
    save(PDF("$outfile.pdf"), t)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
