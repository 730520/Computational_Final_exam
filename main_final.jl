
## First Model----------------------------------------------------------------------------------------###
# loading required packages ----------------------------------------------------------------------------------- #
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
using Plots
using Missings

addprocs(7)
@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
end

using Flux
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

# Defining data sets path---------------------------------------------------------------------------------------##

dir=@__DIR__
cd(dir)

# Read and combine calibration and competition data
train_data = CSV.read("All estimation raw data.csv", DataFrame)

test_data = CSV.read("raw-comp-set-data-Track-2.csv", DataFrame)

raw_data = vcat(train_data, test_data)

# Preprocess data: drop unnecessary variables and clean categorical variables
raw_data = select(raw_data, Not([:SubjID, :RT]))
rename!(raw_data, :B => :choice)

# Clean categorical variables like Location, Gender, Condition, LotShapeA, LotShapeB, and Button
# Mapping Location to numeric values
raw_data.Location .= ifelse.(raw_data.Location .== "Rehovot", 1,
                             ifelse.(raw_data.Location .== "Technion", 2, missing))

# Mapping Gender to numeric values
raw_data.Gender .= ifelse.(raw_data.Gender .== "F", 1,
                           ifelse.(raw_data.Gender .== "M", 2, missing))

# Mapping Condition to numeric values
raw_data.Condition .= ifelse.(raw_data.Condition .== "ByFB", 1,
                               ifelse.(raw_data.Condition .== "ByProb", 2, missing))

# Mapping LotShapeA to numeric values
raw_data.LotShapeA .= ifelse.(raw_data.LotShapeA .== "-", 1,
                                ifelse.(raw_data.LotShapeA .== "L-skew", 2,
                                    ifelse.(raw_data.LotShapeA .== "Symm", 3,
                                        ifelse.(raw_data.LotShapeA .== "R-skew", 4, missing))))

# Mapping LotShapeB to numeric values
raw_data.LotShapeB .= ifelse.(raw_data.LotShapeB .== "-", 1,
                                ifelse.(raw_data.LotShapeB .== "L-skew", 2,
                                    ifelse.(raw_data.LotShapeB .== "Symm", 3,
                                        ifelse.(raw_data.LotShapeB .== "R-skew", 4, missing))))

# Mapping Button to numeric values
raw_data.Button .= ifelse.(raw_data.Button .== "R", 1,
                              ifelse.(raw_data.Button .== "L", 2, missing))


# Define model type
@everywhere model = "RUM"   

# Define hyperparameters
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

# Split data into training and test sets
train_df= collect(1:510750)
test_df = collect(510751:514500)

# Function to process data and return training and test data
function get_processed_data(args)
    # Process features and labels
    labels = string.(raw_data.choice)
    features = Matrix(raw_data[:, 2:end])'
    normed_features = normalise(features, dims=2)
    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)
    
    # Split into training and test sets
    X_train = normed_features[:, train_df]
    y_train = onehot_labels[:, train_df]
    X_test = normed_features[:, test_df]
    y_test = onehot_labels[:, test_df]

    # Repeat training data based on args.repeat
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test, y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 27 features as inputs and outputting 2 probabiltiies
    model = Chain(
    Dense(27, 28, relu),  # Update input size to 27
    Dense(28, 35, relu),
    Dense(35, 35, relu),
    Dense(35, 28, relu),
    Dense(28, 25, relu),
    Dense(25, 20, relu),
    Dense(20, 10),
    softmax,
    Dense(10, 2))
 

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end


# Function to test the model
function test(model, test)
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")
    @assert accuracy_score > 0.1

    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))

    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test, y_test))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data) ## loss= 0.0618 ##

### ------------------------------------------------------------------------------------------------------------------------##

### Second Model:

# Calculate expected values for lotteries A and B
raw_data.ExA = raw_data.Ha .* raw_data.pHa .+ raw_data.La .* (1 .- raw_data.pHa)
raw_data.ExB = raw_data.Hb .* raw_data.pHb .+ raw_data.Lb .* (1 .- raw_data.pHb)

# Assign 1 to ExB_best if its expected value is higher than ExA, else assign 0
raw_data.ExB_best .= ifelse.(raw_data.ExB .> raw_data.ExA, 1, 0)

# Calculate expected values for lotteries HA and HB
raw_data.ExHA = raw_data.Ha .* raw_data.pHa
raw_data.ExHB = raw_data.Hb .* raw_data.pHb

# Assign 1 to ExHB_best if its expected value is higher than ExHA, else assign 0
raw_data.ExHB_best .= ifelse.(raw_data.ExHB .> raw_data.ExHA, 1, 0)



# Define a modified training function with additional inputs in the dense part
function train(; kws...)
    args = Args(; kws...)

    train_data, test_data = get_processed_data(args)

    # Define a neural network model with multiple layers
    model = Chain(
        Dense(33, 33, relu),
        Dense(33, 30, relu),
        Dense(30, 25, relu),
        Dense(25, 20, relu),
        Dense(20, 10),
        softmax,
        Dense(10, 2)
    )

    # Define the loss function using logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)
### loss is 0.0263

##-----------------------------------------------------------------------------------------------------------------------------------##


#### Model 3: Model with attention varibles: 
attention = CSV.read("final_data.csv", DataFrame)
attention = attention[:, 2:end] 
data = hcat(raw_data, attention, makeunique=true)


function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 40 features as inputs and outputting 2 probabiltiies
    #model = Chain(Dense(40, 2))
    model = Chain(
        Dense(33,40,relu),
        Dense(40,40,relu),
        Dense(40,35,relu),
        Dense(35, 30, relu),
        Dense(30, 20, relu),
        Dense(20, 10),
        softmax,
        Dense(10,2))


    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)
### loss is 0.0269


#--------------------------------------------------------------------------------------------------------------------------------------------##

### Extra Model:
#meanRT Moretime Forgone Condition block Trial Payoff Gender GameID Age RT
#Location B_more Order meanB meanB_subject Set Feedback consistent Ha

cols = [:meanRT, :Moretime, :Forgone, :Condition, :block, :Trial, :Payoff, :Gender, :GameID, :Age,  :Location, :Order, :meanB, :meanB_subject, :Set, :Feedback, :consistent, :Ha]
df = select(data, Not(cols))



function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

  
    model = Chain(
        Dense(33, 25, relu),
        Dense(25, 20, relu),
        Dense(20, 20, relu),
        Dense(20, 10, relu),
        Dense(10, 2),
        softmax)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end


cd(@__DIR__)
model, test_data = train()
test(model, test_data)
### loss is 0.313
