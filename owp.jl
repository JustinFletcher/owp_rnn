

# TITLE: Experimental Working File for OWP RNN Training
# DATE: 29 May 16
# VERSION: 0.9
# PROJECT: OWP_RNN
# AUTHOR: Justin Fletcher
# PROCESS DESCRIPTION: This file contains numerous fucntions an processes which
#                      train an RNN using the OWP. Several function in this File
#                      may be used independently, to solve the OWP.
# REQUIRED ADT: GraphTheory::Graph
# LANGUAGE: Julia 0.4


workspace()
("\\csce-686\\project")
include("$(pwd())\\"*"GraphTheory.jl")
include("$(pwd())\\"*"ExperimentDataset.jl")
using GraphTheory

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

function solve_owp_sa(G, u, v, L, v_err)

    ω = ω(G)

    # construct an intial random feasible solution
    # Initialize a sequence.
  	W = Any[]

    #Start with u.
  	push!(W, u)

    #Build vertex label set.
  	V = V(G)

    # Iterate over all but the final walk step.
    for (l in 2:L)

        # If it's the next-to-last walk step...
        if (l==L)
            # Select any vertex but the previous one or v.
            x = randchoice(remove!(remove!(V, W[l-1]), v))
        else
            # Select any vertex but the previous one.
            x = randchoice(remove!(V, W[l-1]))
        end

        # Add x.
        push!(W, x)
    end

    # As a final step, add v.
    push!(W, v)


    # Get the objective function of the random walk.
    # Initialize the weight counter.
    w = 0

    # Iterate over each edge in the walk.
    for i in 1:(length(W)-1)

        # Sum the weights along the path.
        w += ω[W[i], W[i+1]]
    end

    # Inialize the previous walk value.
    w_prev = w

    # Iterate for the specified number of epochs.
    for t in 1:τ

        # Store the current walk.
        W_prev = W

        # Modify the walk.
        mod_locus = randchoice(2:length(W)-1)
        neighbors = remove!(remove!(remove!(W,W[mod_locus-1]),W[mod_locus]),W[mod_locus+1])
        W[mod_locus] = randchoice(remove!(V, neighbors))

        # Initialize the weight counter.
        w = 0

        # Iterate over each edge in the walk.
        for i in 1:(length(W)-1)

            # Sum the weights along the path.
            w += ω[W[i], W[i+1]]
        end



        # Apply the selection criterion to the loss.
        # If this is a downhill move, take it.
        if (w ≤ w_prev)

            w_prev = w

            # If this is not a downhill or neural move.
        else

            # Then with probability exp(-DeltaE/T), reject the move.
            if( (rand() >= exp(-((w -w_prev))/(T₀/(t)))))
                W = W_prev

                # If the uphill move is not rejected, set the error.
            else
                w_prev = w
            end

        end

      end

      return(W)

end

function solve_owp(G, u, v, L, v_err)

    # Parse out the weight matrix.
    ω = w(G)

    # Initialze a stack to store each feasible solution walk.
    W′ = Any[]

    # INITIALIZATION (1): For each of the root walks, find all feasible paths.
    for W₀ in [[u, neighbor] for neighbor in Γ(ω, u)]

        # Initialize a stack.
        S = Any[]

        # Initialize a list of explored walks.
        Wₑ′ = Any[]

        # INITIALIZATION (2): Add this root walk to the stack.
        push!(S, W₀)

        # Iterated until the stack is emptied
        while !(isempty(S))

            # Remove a walk from the stack.
            W = pop!(S)

            # Determine if the walk has been evaluated.
            if !isDiscovered(W, Wₑ′)

                # Add this walk to the evaluated list.
                push!(Wₑ′, copy(W))

                # SOLUTION: Check if W is a solution.
                if ((length(W)==(L+1)) && (W[end]==v))

                    # If it is feasible, add it to the feasible list.
                    push!(W′, copy(W))

                end

                # NEXT-STATE-GENERATION: Generate all augmenting walks from W.
                # (Implicit SELECTION)
                augmenting_walks = [push!(copy(W),neighbor) for neighbor in Γ(ω, W[end])]

                # FEASIBILITY: Select only those walks shorter than L.
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

    # OBJECTIVE (1): Compute the value of each feasible solution.
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

    # OBJECTIVE (2): Select the highest-value walk.
    sort!(weighted_walks, by=function(v) v[2] end, rev=sort_order)

    # Return the complete list of weighted walks.
    return(weighted_walks[1], weighted_walks)

end

# This function executes the recursive propagation of an input pattern.
@everywhere function ϕ(i, ω, α, χ)

    (i==0) ? (return(zerofill(χ, size(ω)[2]))) : (return(tanh(*(ϕ(i-1, ω, α, χ), ω))))

end

# This funciton pads a data set with 0.
function zerofill(v, d)

   return([v  transpose(zeros(d-length(v)))])

end

# this function computes the class error.
function recursiveClassError(ω, X, δ, α, λLength)

    err = 0

    for (χ, λ) in X

        err += ((indmax(λ[1:λLength]) != indmax(ϕ(δ, ω, α, χ)[1:λLength])))

    end

    return(err)
end

function train_rnn_owp_dfs(ω, τ, X₀, Xᵥ, δ, α, n, fᵥ, B)
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

                # Iterate over each edge in the walk.
                v = opw_walk[1][end]
                for i in 1:(length(opw_walk[1])-1)

                    ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e - 0.00001*abs((μ_error_pattern[v])/(W_weight_sum))


                end
            end
        end

        push!(lossVec, μ_loss)

        push!(errVec, fᵥ(ω, Xᵥ, δ, α, λLength)./length(Xᵥ))

        if t%100==0
            println(t)
            println(errVec[end])
        end

    end

    return(lossVec, errVec, ω)

end


function train_rnn_owp_sa(ω, τ, X₀, Xᵥ, δ, α, n, fᵥ, B)
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
                (owp_solution, weighted_walks) = solve_owp_sa(Graph(ω), u, v, δ, μ_error_pattern[v])

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

                # Iterate over each edge in the walk.
                v = opw_walk[1][end]
                for i in 1:(length(opw_walk[1])-1)

                    ω[opw_walk[1][i], opw_walk[1][i+1]] = w_e - 0.00001*abs((μ_error_pattern[v])/(W_weight_sum))


                end
            end
        end

        push!(lossVec, μ_loss)

        push!(errVec, fᵥ(ω, Xᵥ, δ, α, λLength)./length(Xᵥ))

        if t%100==0
            println(t)
            println(errVec[end])
        end

    end

    return(lossVec, errVec, ω)

end




#############

irisDatapath = "$(pwd())\\data\\iris.dat"

dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")


#############################################################################################################

# dataset = lcvfDataset
# This sectino of code contians all that is need to run an isolated instance
# of the OWP-RNN training procedure. Useful for debugging.

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




#############

# In this section, the experimetal procedure is conduted. First, choose the
# desried data set. Then select the desired parameters. Finally, choose the
# desried training algorthim by inserting it into the core of the NFCV loop.


## First, construct the data sets.

# Iris

irisDatapath = "$(pwd())\\data\\iris.dat"

dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")

# Harmonic

function harmonic(x,y)
  return(sin(2.*pi.*sqrt(x.^2+y.^2)))
end

xRange=-1:0.01:1
yRange=-1:0.01:1

figure()
harmonicFuncArray = build2dFunctionArray(harmonic,xRange,yRange)
surf(xRange, yRange, harmonicFuncArray./maximum(abs(harmonicFuncArray)), cmap=ColorMap("coolwarm"),)

xlabel(" \$\ u_1 \$\ ")
ylabel(" \$\ u_2 \$\ ")
zlabel(" \$\ f_{H} \$\ ")

title("Harmonic Function")

harmFunctionSample = sampleFunction(harmonic, 500, xRange, yRange)

figure()

imshow(harmonicFuncArray./maximum(abs(harmonicFuncArray)), cmap=ColorMap("coolwarm"),extent=[minimum(xRange),maximum(xRange),minimum(yRange),maximum(yRange)])

xlabel(" \$\ u_1 \$\ ")
ylabel(" \$\ u_2 \$\ ")

title("HarmonicFunction")

# Cancer

cancerDatapath = "$(pwd())\\data\\wisconsonBreastCancerData.dat"
dataInputDimensions = [1:30]
dataOutputDimensions = [31]

cancerDataset = ExperimentDataset.Dataset(cancerDatapath, dataInputDimensions, dataOutputDimensions, "Cancer")





# Select the desired data set.
dataset = irisDataset


###### n_Fold Cross Validation #####


n = 5

δ = 3

fᵥ = recursiveClassError

α = null

n_folds = 10

T₀ = (n_folds-1)*ifloor(size(dataset.data)[1]/n_folds)

τ = 1*T₀

n_reps = 30

cost_surface=null
# Determine the number of events.
num_events = n_folds*n_reps

# Initilize the storage matrices.
errMatrix = zeros(num_events,τ)
lossMatrix =  zeros(num_events,τ)

# Initilize the event count.
event_count = 0

# Iterate over each fold.
for fold in 0:(n_folds-1)

    # Calculate the fold size.
    fold_size = ifloor(size(dataset.data)[1]/n_folds)

    # Determine the indeces of the validation set for this fold.
    val_set_inds =  ((fold*fold_size)+1):((fold+1)*fold_size)

    # Determine the indeces of the training set for this fold.
    training_set_inds = IntSet(1:size(dataset.data)[1])
    [delete!(training_set_inds, i) for i in val_set_inds]

    # Construct the trianing set.
    X₀ = Any[]
    for rowIndex in training_set_inds
        push!(X₀, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
    end

    # Select the batch to be the complete training set.
    B = length(X₀)

    # Construct the validation set.
    Xᵥ = Any[]
    for rowIndex in val_set_inds
        push!(Xᵥ, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
    end

    # Repeat for the desired number of reps.
    for rep in 1:n_reps

        # Increment the event count.
        event_count += 1

        window_size = 100

        decay_freq=1000

        # Perform train_rnn_owp.
        output_owp = @time train_rnn_owp(copy(ω), τ, X₀, Xᵥ, δ, α, n, fᵥ, B)

        # Parse the output tuple.
        (lossVec, errVec, ω) = output_owp


        # Add the observed validation error and loss to the matrix.
        errMatrix[event_count, :] = errVec
        lossMatrix[event_count, :] = lossVec


    end

end


################ Plotting

using PyPlot


plot(plot_τ , vec(plot_meanErrVec_sa./2), label="OWP-DFS-BT", color="blue")
errorbar(plot_τ , vec(plot_meanErrVec_sa./2), yerr=vec(plot_stdErrVec_sa./2), fmt=".", alpha=0.7, color="blue")
title("Validation Set Classification Error Through Training ")
xlabel(" \$\ t \$\ (Epoch Number)")
ylabel("  Mean Validation Error ")
legend(loc=2)

#### Now, we run the same experiment using SA




## First, construct the data sets.

# Iris

irisDatapath = "$(pwd())\\data\\iris.dat"

dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")

# Harmonic

function harmonic(x,y)
  return(sin(2.*pi.*sqrt(x.^2+y.^2)))
end

xRange=-1:0.01:1
yRange=-1:0.01:1

figure()
harmonicFuncArray = build2dFunctionArray(harmonic,xRange,yRange)
surf(xRange, yRange, harmonicFuncArray./maximum(abs(harmonicFuncArray)), cmap=ColorMap("coolwarm"),)

xlabel(" \$\ u_1 \$\ ")
ylabel(" \$\ u_2 \$\ ")
zlabel(" \$\ f_{H} \$\ ")

title("Harmonic Function")

harmFunctionSample = sampleFunction(harmonic, 500, xRange, yRange)

figure()

imshow(harmonicFuncArray./maximum(abs(harmonicFuncArray)), cmap=ColorMap("coolwarm"),extent=[minimum(xRange),maximum(xRange),minimum(yRange),maximum(yRange)])

xlabel(" \$\ u_1 \$\ ")
ylabel(" \$\ u_2 \$\ ")

title("HarmonicFunction")

# Cancer

cancerDatapath = "$(pwd())\\data\\wisconsonBreastCancerData.dat"
dataInputDimensions = [1:30]
dataOutputDimensions = [31]

cancerDataset = ExperimentDataset.Dataset(cancerDatapath, dataInputDimensions, dataOutputDimensions, "Cancer")





# Select the desired data set.
dataset = irisDataset


###### n_Fold Cross Validation #####


n = 5

δ = 3

fᵥ = recursiveClassError

α = null

n_folds = 10

T₀ = (n_folds-1)*ifloor(size(dataset.data)[1]/n_folds)

τ = 1*T₀

n_reps = 30

cost_surface=null
# Determine the number of events.
num_events = n_folds*n_reps

# Initilize the storage matrices.
errMatrix = zeros(num_events,τ)
lossMatrix =  zeros(num_events,τ)

# Initilize the event count.
event_count = 0

# Iterate over each fold.
for fold in 0:(n_folds-1)

    # Calculate the fold size.
    fold_size = ifloor(size(dataset.data)[1]/n_folds)

    # Determine the indeces of the validation set for this fold.
    val_set_inds =  ((fold*fold_size)+1):((fold+1)*fold_size)

    # Determine the indeces of the training set for this fold.
    training_set_inds = IntSet(1:size(dataset.data)[1])
    [delete!(training_set_inds, i) for i in val_set_inds]

    # Construct the trianing set.
    X₀ = Any[]
    for rowIndex in training_set_inds
        push!(X₀, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
    end

    # Select the batch to be the complete training set.
    B = length(X₀)

    # Construct the validation set.
    Xᵥ = Any[]
    for rowIndex in val_set_inds
        push!(Xᵥ, ((dataset.data[rowIndex, dataset.inputCols], dataset.data[rowIndex, dataset.outputCols])))
    end

    # Repeat for the desired number of reps.
    for rep in 1:n_reps

        # Increment the event count.
        event_count += 1

        window_size = 100

        decay_freq=1000

        # Perform train_rnn_owp.
        output_owp = @time train_rnn_owp_sa(copy(ω), τ, X₀, Xᵥ, δ, α, n, fᵥ, B)

        # Parse the output tuple.
        (lossVec, errVec, ω) = output_owp


        # Add the observed validation error and loss to the matrix.
        errMatrix[event_count, :] = errVec
        lossMatrix[event_count, :] = lossVec


    end

end


################ Plotting

using PyPlot

plot(plot_τ , vec(plot_meanErrVec_sa./2), label="OWP-SA", color="blue")
errorbar(plot_τ , vec(plot_meanErrVec_sa./2), yerr=vec(plot_stdErrVec_sa./2), fmt=".", alpha=0.7, color="blue")
title("Validation Set Classification Error Through Training ")
xlabel(" \$\ t \$\ (Epoch Number)")
ylabel("  Mean Validation Error ")
legend(loc=2)
