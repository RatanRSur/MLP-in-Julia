stod(str::String) = convert(Float64,parse(str))::Float64
stoi(str::String) = int(parse(str))::Int
line2Floats(str::String) = [stod(x) for x in split(str::String)]::Array{Float64}
line2Ints(str::String) = [stoi(x) for x in split(str::String)]::Array{Int}

function loadInit(init::IO)
        initLines=readlines(init)
        (ni, nh, no) = int(split(initLines[1]))
        Theta1 = zeros(nh,ni+1)
        for i=1:nh
                Theta1[i,:] = line2Floats(initLines[1+i])
        end
        Theta2 = zeros(no, nh+1)
        for i=1:no
                Theta2[i,:] = line2Floats(initLines[1+nh+i])
        end

        return ni, nh, no, Theta1, Theta2
end

function loadData(data::IO)
        trainLines=readlines(data)
        (m, ni, no) = int(split(trainLines[1]))
        X = zeros(m, ni)
        y = zeros(Int, m, no)
        for i=1:m
               X[i,:] = line2Floats(trainLines[1+i])[1:ni]
               y[i,:] = line2Ints(trainLines[1+i])[ni+1:end]
        end
        return X::Array{Float64}, y::Array{Int}
end
