## Q2

## Part i

require(ggplot2)
trSet_ <- read.csv("./assessments_datasets/Task3B_train.csv")
tsSet_ <- read.csv("./assessments_datasets/Task3B_test.csv")

ggplot(data = trSet_, aes(x = trSet_$x1, y = trSet_$x2)) + geom_point(aes(colour = as.factor(trSet_$y))) + ggtitle("Training Set Data") + ylab("X2") + xlab("X1") + labs(colour = "Cluster");



# part ii

trSet_ <- trSet_[complete.cases(trSet_),];
tsSet_ <- tsSet_[complete.cases(tsSet_),];


train.data <- trSet_[, 1:2];
train.label <- trSet_$y;
test.data <- tsSet_[, 1:2];
test.label <- tsSet_$y;
train.len <- nrow(train.data);

A31Perceptron <- function(train.data, train.label, test.data, test.label) {

    # This is a wrapped implementation of the perceptron provided in activity 3.1
    # It is copied and pasted. The only change is highlighted below.
    #
    # This simply brings in the class label naming and 


    ########################################################################
    c0 <- 1;
    c1 <- -1; # class labels

    train.len <- nrow(train.data);
    train.index <- sample(1:train.len, train.len, replace = FALSE);
    #########################################################################



    ## Basis function (Step 1)
    Phi <- as.matrix(cbind(1, train.data)) # add a column of 1 as phi_0

    # Initialization
    eta <- 0.01 # Learning rate
    epsilon <- 0.001 # Stoping criterion
    tau.max <- 100 # Maximum number of iterations

    T <- ifelse(train.label == c0, eval(parse(text = c0)), eval(parse(text = c1))) # Convention for class labels

    W <- matrix(, nrow = tau.max, ncol = ncol(Phi)) # Empty Weight vector
    W[1,] <- runif(ncol(Phi)) # Random initial values for weight vector

    error.trace <- matrix(0, nrow = tau.max, ncol = 1) # Placeholder for errors
    error.trace[1] <- sum((Phi %*% W[1,]) * T < 0) / train.len * 100 # record error for initial weights

    tau <- 1 # iteration counter 
    terminate <- FALSE # termination status

    # Main Loop (Step 2):
    while (!terminate) {
        # resuffling train data and associated labels:
        train.index <- sample(1:train.len, replace = FALSE)
        Phi <- Phi[train.index,]
        T <- T[train.index]

        for (i in 1:train.len) {
            if (tau == tau.max) {
                break
            }

            # look for missclassified samples
            if ((W[tau, ] %*% Phi[i, ]) * T[i] < 0) {

                # update tau counter
                tau <- tau + 1

                # update the weights
                W[tau,] <- W[tau - 1,] + eta * Phi[i,] * T[i]

                # update the records
                error.trace[tau] <- sum((Phi %*% W[tau,]) * T < 0) / train.len * 100
            }

        }

        # decrease eta:
        eta = eta * 0.99
        # recalculate termination conditions
        terminate <- tau >= tau.max |
        abs(sum((Phi %*% W[tau,]) * T < 0) / train.len - sum((Phi %*% W[tau - 1,]) * T < 0) / train.len) <= epsilon

    }
    W <- W[1:tau,] # cut the empty part of the matrix (when the loop stops before tau == tau.max)

    ## the  final result is w:
    w <- W[tau,]


    return(list("err" = error.trace, "coeff" = w));
}


res_ <- A31Perceptron(train.data, train.label, test.data, test.label)


A31Predict <- function(data,labels,params,basis=1) {

    # wrapper function for producing predictions based on some weights.

    predictions_ <- as.matrix(cbind(basis,data)) %*% as.matrix(params)

    predictions_ <- ifelse(predictions_ < 0, 0, 1);
    

    return(predictions_);

}


preds_ <- A31Predict(test.data, test.label, res_$coeff)
preds_ <- cbind(train.data, preds_);

colnames(preds_) <- c("x1", "x2", "y");

ggplot(data = preds_, aes(x = preds_$x1, y = preds_$x2)) + geom_point(aes(colour = as.factor(preds_$y))) + ggtitle("Training Set Data") + ylab("X2") + xlab("X1") + labs(colour = "Cluster");