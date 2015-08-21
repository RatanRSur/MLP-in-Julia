include("loadTrain.jl")
include("mathNN.jl")
include("saveNN.jl")

(init, train, output, nepochs, alpha) = promptTrain()
(ni, nh, no, Theta1, Theta2) = loadInit(init)
const (X, y) = loadData(train)

while nepochs>0
        (J, Theta1, Theta2) = gradientDescent(costFunction, Theta1, Theta2, X, y, alpha)
        nepochs -= 1
        @printf("\rEpochs remaining: %i\t Cost: %.4f", nepochs, J)
end
println("\rTraining Complete      ")

write(output, "$ni $nh $no\n")
writeThetas(output, Theta1, Theta2)
close(output)
