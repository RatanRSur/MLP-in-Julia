require("argparse")
using ArgParse

include("load.jl")

function parseTrain()
        s = ArgParseSettings()
        @add_arg_table s begin
                "init"
                        help = "init file"
                        required = false
                "train"
                        help = "train file"
                        required = false
                "output"
                        help = "output file"
                        required = false
                "-e"
                        help = "number of epochs"
                        arg_type=Int
                "-a"
                        help = "learning rate"
                        arg_type=Float64
        end

        return parse_args(s)
end

function promptTrain()
        parsedArgs=parseTrain()

        if parsedArgs["init"] == nothing
                println("File containing neural network initialization:")
                initName = readline(STDIN)[1:end-1]
        else
                initName = parsedArgs["init"]
        end

        if parsedArgs["train"] == nothing
                println("File containing training set:")
                trainName = readline(STDIN)[1:end-1]
        else
                trainName = parsedArgs["train"]
        end

        if parsedArgs["output"] == nothing
                println("Name of output file:")
                outputName = readline(STDIN)[1:end-1]
        else
                outputName = parsedArgs["output"]
        end

        if parsedArgs["e"] == nothing
                println("Number of epochs:")
                nepochs = int(readline(STDIN))
        else
                nepochs = parsedArgs["e"]
        end

        # if parsedArgs["a"] == nothing
                println("Learning rate, alpha:")
                alpha = double(readline(STDIN))
        else
                alpha = parsedArgs["a"]
        end

        init = open(initName)
        train = open(trainName)
        output = open(outputName, "w")

        return init, train, output, nepochs, alpha
end


function loadTrainingData(train::IO)
        trainLines=readlines(train)
        (m, ni, no) = int(split(trainLines[1]))
        X = zeros(m, ni)
        y = zeros(m, no)
        for i=1:m
               X[i,:] = line2Arr(trainLines[1+i])[1:ni]
               y[i,:] = line2Arr(trainLines[1+i])[ni+1:end]
        end
        return X, y
end
