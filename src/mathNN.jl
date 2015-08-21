sigmoid(z) = 1./(1+exp(-z))

function sigprime(z)
        s = sigmoid(z)
        return s .* (1-s)
end

addBias(matr) = [-ones(size(matr,1),1) matr]

function feedForward(input, theta)
        z = addBias(input)* theta'
        return z, sigmoid(z)
end


function costFunction(Theta1, Theta2, X, y)
        m=size(X,1)
        z2, a2 = feedForward(X,Theta1)
        z3, a3 = feedForward(a2, Theta2)
        J = (1/m) * sum(sum(-y.*log(a3) - (1-y).*log(1-a3)))

        Delta1=zeros(size(Theta1))
        Delta2=zeros(size(Theta2))
        for i=1:m
                d3 = a3[i,:]- y[i,:] #row
                Delta2 += d3' * addBias(a2[i,:])
                d2 = Theta2'[2:end,:] * d3' .* sigprime(z2[i,:]')
                Delta1 += d2 * addBias(X[i,:])
        end
        Delta1 ./= m
        Delta2 ./= m
        return J, Delta1, Delta2
end

function gradientDescent(costFunc, Theta1, Theta2, X, y, alpha)
        (J, Delta1, Delta2) = costFunc(Theta1, Theta2, X, y)
        Theta1 -= alpha * Delta1
        Theta2 -= alpha * Delta2
        return J, Theta1, Theta2
end

function gradientChecking(costFunc, Theta1, Theta2, X, y)
end

function predict(X, Theta1, Theta2)
        return int(feedForward(feedForward(X,Theta1)[2],Theta2)[2])
end

function contingencyTable(pred, y)
        tpMatr = pred & y
        tnMatr = pred.==0 & y.==0
        fpMatr = (pred.==1) & (y .==0)
        fnMatr = (pred.==0) & (y .==1)
        return [sum(tpMatr, 1) sum(tnMatr, 1) sum(fpMatr, 1) sum(fnMatr, 1)]
end

function accuracy(tp,tn,fp,fn)
        (tp + tn)/(tp + tn + fp + fn)
end

precision(tp, fp) = tp/(tp+fp)

recall(tp, fn) = tp/(tp+fn)

F1(prec, rec) = 2*prec*rec/(prec+rec)

function evalutaions(pred,y)
        contingencyTable
        acc = accuracy(tp, tn, fp, fn)
        prec = precision(tp,fp)
        rec = recall(tp,fn)
        f1 = F1(prec, rec)
        return tp, tn, fp, fn, acc, prec, rec, f1
end
