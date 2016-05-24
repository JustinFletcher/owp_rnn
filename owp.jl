workspace()
cd("\\csce-686\\project")
include("$(pwd())\\"*"GraphTheory.jl")
include("$(pwd())\\"*"ExperimentDataset.jl")
using GraphTheory

rmprocs()

function Γ(ω, u)

    # This function returns the vertices adjacent to u. In the
    # case of a digraph, it returns the vertices for which u
    # is the tail of an incident edge.

    # Determine outgoing edges.
    a = ω[u,:].!=0
    o = Int64[]

    # Build a vector of the vertex IDs for which u is a tail.
    for i = 1:length(ω[1,:])
        if a[i]
            push!(o, i)
        end
    end
    return(o)

end

function isDiscovered(W, Wₑ′)

    # Iterate over each discovered walk.
    for Wₑ in Wₑ′

        # If the given walk has been explored...
        if W == Wₑ

            # Return true.
            return(true)

        end
    end

    # If you get here, the walk has not been explored.
    return(false)

end

function solve_owp(G, u, v, L, v_err)

    # Parse out the weight matrix.
    ω = w(G)

    # Initialze a stack to store each feasible solution walk.
    W′ = Any[]

    # For each of the root walks, find all feasible paths.
    for W₀ in [[u, neighbor] for neighbor in Γ(ω, u)]

        # Initialize a stack.
        S = Any[]

        # Initialize a list of explored walks.
        Wₑ′ = Any[]

        # Add this root walk to the stack.
        push!(S, W₀)

        # Iterated until the stack is emptied
        while !(isempty(S))

            # Remove a walk from the stack.
            W = pop!(S)

            # Determine if the walk has been evaluated.
            if !isDiscovered(W, Wₑ′)

                # Add this walk to the evaluated list.
                push!(Wₑ′, copy(W))

                # Check if W is a feasible solution.
                if ((length(W)==(L+1)) && (W[end]==v))

                    # If it is feasible, add it to the feasible list.
                    push!(W′, copy(W))

                end

                # Generate all augmenting walks from W.
                augmenting_walks = [push!(copy(W),neighbor) for neighbor in Γ(ω, W[end])]

                # Select only thos walks shorter than L.
                feasible_walks = Any[]
                for augmenting_walk in augmenting_walks
                    if (length(augmenting_walk) <= (L+1))
                        push!(feasible_walks, augmenting_walk)
                    end
                end

                # Add each of the child nodes (walks) to S.
                [push!(S, feasible_walk) for feasible_walk in feasible_walks]

            end
        end
    end

    # Replace this entire structure with a Walk type.
    # Initialize the weighted walks vector.
    weighted_walks = Any[]

    # Compute the value of each feasible solution.
    for W in W′

        # Initialize the weight counter.
        w = 0

        # Iterate over each edge in the walk.
        for i in 1:(length(W)-1)

            # Sum the weights along the path.
            w += ω[W[i], W[i+1]]
        end

        # Add the weight tuple, to the list.
        push!(weighted_walks,(W, w))

    end

    # Select the sort order.
    sort_order = (v_err>0)

    # Select the highest-value walk.
    sort!(weighted_walks, by=function(v) v[2] end, rev=sort_order)

    # Return the complete list of weighted walks.
    return(weighted_walks[1], weighted_walks)

end

n = 15
L = 3
(owp_solution, weighted_walks) = @time solve_owp(K(n, randn), 1, 2, L, +1)

weighted_walks
owp_solution

@everywhere function ϕ(i, ω, α, χ)

    (i==0) ? (return(zerofill(χ, size(ω)[2]))) : (return(tanh(*(ϕ(i-1, ω, α, χ), ω))))

end

function zerofill(v, d)

   return([v  transpose(zeros(d-length(v)))])

end

function recursiveClassError(ω, X, δ, α, λLength)

    err = 0

    for (χ, λ) in X

        err += ((indmax(λ[1:λLength]) != indmax(ϕ(δ, ω, α, χ)[1:λLength])))

    end

    return(err)
end

function train_rnn_owp(ω, τ, X₀, Xᵥ, δ, α, n, fᵥ, B)
    #  O(τ|X|δn²)

    ≤ = <=

    # Initialize the loss tracking vector.
    lossVec = Float64[]

    # Initialize the validation error vector.
    errVec = Float64[]

    # Store the implicit length of λ.
    λLength = size(X₀[1][2])[2]

    # Store the implicit length of λ.
    χLength = size(X₀[1][1])[2]

    # Precondition X for propagation.
    X = [(zerofill(χ, n), zerofill(λ, n)) for (χ, λ) in X₀]

    # Precondition Xᵥ for propagation.
    Xᵥ = [(zerofill(χ, n), zerofill(λ, n)) for (χ, λ) in Xᵥ]

    # Initialize the loss.
    loss = Inf

    # Iterate for the specified number of epochs. O(τ|X|δn²)
    for t in 1:τ

        # Randomly select a batch B=10
        b = X₀[randperm(length(X₀))[1:B]]

        # Initialize error vector accumulator.
        Σ_error_pattern = zeros(λLength)

        # Loss sum.
        Σ_loss = 0

        # Iterate over each data-output pair. O(|X|δn²)
        for (χ, λ) in b

            # Propogate the signal through the network. O(δn²)
            error_pattern = (ϕ(δ, ω, α, χ)[1:λLength]-λ[1:λLength])

            loss = *(transpose(error_pattern),error_pattern)[1]

            Σ_loss += loss

            Σ_error_pattern += error_pattern

        end
        μ_loss = Σ_loss/length(b)

        μ_error_pattern = Σ_error_pattern
#         μ_error_pattern = Σ_error_pattern./sum(Σ_error_pattern)
        println(μ_error_pattern)
        μ_error_pattern = Σ_error_pattern
        # Iterate over each output neuron.
        for v in 1:λLength

            # Initialize a set to hold the optimal paths to v.
            W = Any[]

            # Iterate over each input neuron.
            for u in 1:χLength

                # Find the optimal walk problem.
                (owp_solution, weighted_walks) = solve_owp(Graph(ω), u, v, δ, μ_error_pattern[v])

                # Add the optimal path from u to the list.
                push!(W, owp_solution)

            end


            # Compute the sum of weights along all optimal u,v walks.
            W_weight_sum = 0
            for opw_walk in W
                W_weight_sum += opw_walk[2]
            end


            # Apply the weight update rule to each weights in the optimal paths.
            for opw_walk in W

#                 println("---For Walk: $opw_walk")
                # Iterate over each edge in the walk.

                v = opw_walk[1][end]

                for i in 1:(length(opw_walk[1])-1)

#                     println("------Edge: {$(opw_walk[1][i]),$(opw_walk[1][i+1])}")

                    # Compute the weight update.
#                     update = (1+((μ_error_pattern[v]*opw_walk[2])/(W_weight_sum)))
#                     update = (1-((μ_error_pattern[v])/(W_weight_sum)))

#                     println("Update Value = $update")
#                     println("Error Value = $(μ_error_pattern)")

#                     p = ((μ_error_pattern[v]*opw_walk[2])/(W_weight_sum))

#                     p = ((μ_error_pattern[v])/(W_weight_sum))


#                     println("p Value = $p")
                    # Sum the weights along the path.

#                     ω[opw_walk[1][i], opw_walk[1][i+1]] = ω[opw_walk[1][i], opw_walk[1][i+1]]*update
                    w_e = ω[opw_walk[1][i], opw_walk[1][i+1]]

#                     if(sign(w_e)==sign(μ_error_pattern[v]))
#                         ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e - 0.001w_e*((μ_error_pattern[v])/(W_weight_sum))
#                     else
#                         ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e + 0.001w_e*((μ_error_pattern[v])/(W_weight_sum))
#                     end
                    p = abs((opw_walk[2])/(W_weight_sum))

                    if(μ_error_pattern[v]>0)
                        ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e - p
                    else
                        ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e + p
                    end

#                     println("perturbation = $(p)")

#                     ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e - 0.00001*abs((μ_error_pattern[v])/(W_weight_sum))


                end
            end
        end

#         println(weighted_walks)

#         push!(errVec, fᵥ(ω, Xᵥ, δ, α, λLength))
        push!(lossVec, μ_loss)

        push!(errVec, fᵥ(ω, Xᵥ, δ, α, λLength)./length(Xᵥ))
#         push!(errVec, fᵥ(ω, X₀, δ, α, λLength)./length(X₀))

        if t%100==0
            println(t)
            println(errVec[end])
        end

    end

    return(lossVec, errVec, ω)

end

irisDatapath = "$(pwd())\\data\\iris.dat"

dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")

###############

# lcvfDatapath = "$(pwd())\\data\\lcvfData.csv"
# dataInputDimensions = [1:194]
# dataOutputDimensions = [195]

# lcvfDataset = ExperimentDataset.Dataset(lcvfDatapath, dataInputDimensions, dataOutputDimensions, "LCVF")

#############################################################################################################

# dataset = lcvfDataset

portionX₀ = 0.5

dataset = irisDataset

partitionX₀ = ifloor(size(dataset.data)[1]*portionX₀)

X₀ = Any[]
for rowIndex in 1:partitionX₀
    push!(X₀, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
end

Xᵥ = Any[]
for rowIndex in (partitionX₀+1):size(dataset.data)[1]
    push!(Xᵥ, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
end

n = 5

δ = 3

τ = 200

B = length(X₀)

fᵥ = recursiveClassError

α = null

# Initilize the inverse identity mask.
invIndentityMask = int(!bool(eye(n,n)))

# Initialize ω."
ω = 0.01 .* (randn((n,n)) .* invIndentityMask)

println("Done loading.")

output_owp= @time train_rnn_owp(copy(ω), τ, X₀, Xᵥ, δ, α, n, fᵥ, B)

(lossVec, errVec, ωₒ) = output_owp

# putdata(output_owp,"owp")
# getdata("MatrixProp_CSA_LCVF")


################ Plotting

using PyPlot

subplot(2,2,1)
plot(1:τ , vec(lossVec), label="Loss", color="blue")
# errorbar(1:τ , vec(scp_runtime_by_nsets_mat_mean), yerr=vec(scp_runtime_by_nsets_mat_std), fmt=".", alpha=0.7, color="blue")
title("Loss Through Training")
xlabel(" \$\ t \$\ (Epoch Number)")
ylabel("  Loss  ")
legend(loc=2)

subplot(2,2,3)
plot(1:τ , vec(errVec), label="Validation Set Clssification Error", color="blue")
# errorbar(1:τ , vec(scp_runtime_by_nsets_mat_mean), yerr=vec(scp_runtime_by_nsets_mat_std), fmt=".", alpha=0.7, color="blue")
title("Validation Set Classification Error Through Training ")
xlabel(" \$\ t \$\ (Epoch Number)")
ylabel("  Loss  ")
legend(loc=2)
ylim(0,1)

subplot(2,2,(2,4))
imshow(ωₒ, interpolation="none")
colorbar()
