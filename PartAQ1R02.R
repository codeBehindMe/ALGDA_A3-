# Expectation maximisation

EM_CountFeatureVector <- function(cfv, K = 4,hard=FALSE,eps=1e-10) {

    # this function takes a CountFeatureVector from text preprocessing and clusters documents using Expectation Maximisation.

    logSum <- function(v) {
        ## --- helper function ------------------------------------------------------------------ 
        # Input:    logA1, logA2 ... logAn
        # Output:   log(A1+A2+...+An)
        #
        # This function is needed to prevent numerical overflow/underflow when working with small numbers, 
        # because we can easily get small numbers by multiplying p1 * p2 * ... * pn (where 0 <= pi <= 1 are probabilities).   
        #
        # Example: Suppose we are interested in p1*p2*p3 + q1*q2+q3 where all numbers are probabilities \in [0,1]
        #          To prevent numerical errors, we do the computation in the log space and convert the result back using the exp function 
        #          Hence our approach is to form the vector v = [log(p1)+log(p2)+log(p3) , log(q1)+log(q2)+log(q3)] 
        #          Then get the results by: exp(logSum(v))

        m = max(v)
        return(m + log(sum(exp(v - m))))
    }


    # get the number of documents
    nDocs_ <- dim(cfv)[2] # since document will be given columnwise.
    nWords_ <- dim(cfv)[1] # unique words, given rowwise.


    ################ INITIALISATION #############################


    # initialise theta. (parameters)
    phi_ <- matrix(1 / K, nrow = K, ncol = 1) # this is the mixing component. Assume that all clusters have equal datapoints assigned to it.
    mu_ <- matrix(runif(K * nWords_), nrow = K, ncol = nWords_) # initialise the cluster centers for each word (colwise) and earch cluster (rowwise).
    mu_ <- prop.table(mu_, margin = 1) # normalization to ensure that sum of each row is 1

    # container for params
    theta_ <- list("phi" = phi_, "mu" = mu_);

    # create an empty matrix for the posterior estimates, which is p(Z|X,thetaOld) ln p(X,Z|theta) number of training data samples long.
    gamma_ <- matrix(0, nrow = nDocs_, ncol = K);


    ################# END INITIALISATION #######################

    ############### BEGIN E.STEP #########################

    for (n in 1:nDocs_) {

        for (k in 1:K) {
            ## calculate the posterior based on the estimated mu and rho in the "log space"
            gamma_[n, k] <- log(theta_$phi[k, 1]) + sum(cfv[, n] * log(theta_$mu[k,]))

        }
        # normalisation to sum to 1 in the log space
        logZ = logSum(gamma_[n,])
        gamma_[n,] = gamma_[n,] - logZ
    }

    # reconvert from logspace
    gamma_ <- exp(gamma_);

    # if hard cluster make max gamma 1 and else 0
    if (hard) {
        maxGamma_ <- gamma_ == apply(gamma_, 1, max) # find cluster with max probl.
        gamma_[maxGamma_] <- 1 # set max prob cluster to 1
        gamma_[!maxGamma_] <- 0 # 0 for all others.
    }

    ############# END E.STEP #######################


    ############### BEGIN M.STEP ######################
    for (k in 1:K) {
        # update the mixing components
        phi_[k] <- sum(gamma_[, k]) / nDocs_ # relative cluster size.

        for (n in 1:nWords_) {

            # update the parameters. 
            mu_[k, n] <- sum(cfv[n,gamma_[k]])/sum(cfv[n,])

        }

    }

}