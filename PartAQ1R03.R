read.data <- function(file.name = './assessments_datasets/Task3A.txt', sample.size = 1000, seed = 100, pre.proc = TRUE, spr.ratio = 0.90) {
    # INPUTS:
    ## file.name: name of the input .txt file
    ## sample.size: if == 0  reads all docs, otherwise only reads a subset of the corpus
    ## seed: random seed for sampling (read above)
    ## pre.proc: if TRUE performs the preprocessing (recommended)
    ## spr.ratio: is used to reduce the sparcity of data by removing very infrequent words
    # OUTPUTS:
    ## docs: the unlabled corpus (each row is a document)
    ## word.doc.mat: the count matrix (each rows and columns corresponds to words and documents, respectively)
    ## label: the real cluster labels (will be used in visualization/validation and not for clustering)

    # Read the data
    text <- readLines(file.name)
    # select a subset of data if sample.size > 0
    if (sample.size > 0) {
        set.seed(seed)
        text <- text[sample(length(text), sample.size)]
    }
    ## the terms before the first '\t' are the lables (the newsgroup names) and all the remaining text after '\t' are the actual documents
    docs <- strsplit(text, '\t')
    # store the labels for evaluation
    labels <- unlist(lapply(docs, function(x) x[1]))
    # store the unlabeled texts    
    docs <- data.frame(unlist(lapply(docs, function(x) x[2])))


    library(tm)
    # create a corpus
    docs <- DataframeSource(docs)
    corp <- Corpus(docs)

    # Preprocessing:
    if (pre.proc) {
        corp <- tm_map(corp, removeWords, stopwords("english")) # remove stop words (the most common word in a language that can be find in any document)
        corp <- tm_map(corp, removePunctuation) # remove pnctuation
        corp <- tm_map(corp, stemDocument) # perform stemming (reducing inflected and derived words to their root form)
        corp <- tm_map(corp, removeNumbers) # remove all numbers
        corp <- tm_map(corp, stripWhitespace) # remove redundant spaces 
    }
    # Create a matrix which its rows are the documents and colomns are the words. 
    dtm <- DocumentTermMatrix(corp)
    ## reduce the sparcity of out dtm
    dtm <- removeSparseTerms(dtm, spr.ratio)
    ## convert dtm to a matrix
    word.doc.mat <- t(as.matrix(dtm))

    # Return the result
    return(list("docs" = docs, "word.doc.mat" = word.doc.mat, "labels" = labels))
}

intern_logSum <- function(v) {
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


intern_expectation <- function(featureVector, K, theta, gamma,hard=FALSE) {
    # This internal function implements the expectation step for the EM for GMMs.

    # ARGS #
    ########
    # featureVector ~ Count / TF-IDF vector as matrix.
    # K ~ Number of expected clusters.
    # theta ~ List object container with mixing component "phi" and cluster centroids "mu".
    # gamma ~ posterior estimates for p(Z|X,thetaOld)
    # axis ~ Indicate whether the unique words are given rowwise or column wise. 1 Indicates rowwise. 
    # hard ~ Whether the cluster assignment is probabilistic or absolute. If hard = TRUE, then cluster assignments are made by assignining the cluster of the highest probability of generation. i.e. argmax ln(p(Z|X,theta)).

    # RETURN #
    ##########
    # Returns posterior estmates for theta as matrix.


    # recast feature vector to local variable.
    fv_ <- featureVector;

    # get number of documents and number of unique words
    
    N <- dim(fv_)[2] # number of documents.
    K <- K # number of clusters.


    # BEGIN #
    #########

    for (n in 1:N) {
        # for each document.
        
        for (k in 1:K) {
            # for each cluster.
            ## calculate the posterior based on the estimated mu and rho in the "log space"
            gamma[n, k] <- log(theta$phi[k, 1]) + sum(fv_[, n] * log(theta$mu[k,]))
        }

        # normalise to  1 in the logspace
        logZ_ = intern_logSum(gamma[n,])
        gamma[n,] = gamma[n,] - logZ_

    }

    # convert them back from log
    gamma <- exp(gamma)

    # if hard cluster make max gamma 1 and else 0
    if (hard) {
        maxGamma_ <- gamma == apply(gamma, 1, max) # find cluster with max probl.
        gamma[maxGamma_] <- 1 # set max prob cluster to 1
        gamma[!maxGamma_] <- 0 # 0 for all others.
    }

    return(gamma);

}

intern_maximisation <- function(featureVector,gamma,theta,K) {

    # This internal function implements the expectation step for the EM for GMMs.

    # ARGS #
    ########
    # featureVector ~ Count / TF-IDF vector as matrix.
    # K ~ Number of expected clusters.
    # theta ~ List object container with mixing component "phi" and cluster centroids "mu".
    # gamma ~ posterior estimates for p(Z|X,thetaOld)
    # axis ~ Indicate whether the unique words are given rowwise or column wise. 1 Indicates rowwise. 

    # RETURN #
    ##########
    # Theta with new mu (word proportion parameters) and phi(mixing parameters)

    # recast feature vector to local variable.
    fv_ <- featureVector;

    N <- dim(fv_)[2] # number of documents.
    W <- dim(fv_)[1] # number of words.
    K <- K # number of clusters.

    # BEGIN #
    #########
    for (k in 1:K) {
        # for each K update the new mixing components.
        theta$phi[k] <- sum(gamma[, k] / N);

        # for each document update the word proportion parameter mu_k_w
        for (w in 1:W) {
            # for each word find how many times it occures in the documents belonging to current cluster K.
            theta$mu[k,w] <- sum(gamma[, k] * fv_[w,])/sum(fv_[w,]);

        }
    }

    return(theta);
}

intern_thetaInit <- function(nWords,K=4,seed=1234) {

    # This function initialises theta for document clustering. This includes phi (AKA rho), and mu.

    # ARGS #
    ########

    # nWords ~ Length of number of unique words as integer.
    # K ~ Number of expected clusters as integer.
    # seed ~ Reproducibility, set to NULL if not required.

    # RETURN #
    ##########
    # List object with matrices for phi (mixing components) and mu (word proportion parameters)

    if (!is.null(seed)) {
        set.seed(seed);
    }

    # initialise phi(AKA rho)
    phi_ <- matrix(1 / K, nrow = K, ncol = 1);

    # initalise mu randomly
    mu_ <- matrix(runif(K * nWords), nrow = K, ncol = nWords);
    # normalise mu [0,1]
    mu_ <- prop.table(mu_, 1);


    return(list("phi" = phi_, "mu" = mu_));
}


dClust_eMax <- function(FeatureVector, K = 4,iterMax=10, hard = FALSE,seed=1234) {

    # This function takes a CountFeatureVector from text preprocessing and clusters documents using Expectation Maximisation.
    
    # ARGS #
    ########
    # FeatureVector ~ Count / TF-IDF vector as matrix.
    # K ~ Number of expected clusters.
    # hard ~ Whether the cluster assignment is probabilistic or absolute. If hard = TRUE, then cluster assignments are made by assignining the cluster of the highest probability of generation. i.e. argmax ln(p(Z|X,theta)).
    # axis ~ Indicate whether the unique words are given rowwise or column wise. 1 Indicates rowwise. 



    # RETURN #
    ##########
    # Returns an object containing cluster and other items.


    # recast feature vector to local variable.
    fv_ <- FeatureVector;

    # get number of documents and number of unique words

        nDocs_ <- dim(fv_)[2] # number of documents.
        nWords_ <- dim(fv_)[1] # number of unique words.




    # INITIALISATION #
    ##################

    # initialise parameters using parameter initialisation function.
    theta_ <- intern_thetaInit(nWords = nWords_, K, seed);

    # initialise gamma for posterior probabilities that each document belongs to cluster K. 
    gamma_ <- matrix(0, nrow = nDocs_, ncol = K);


    # MAIN ITERATION #
    ##################

    for (i in 1:iterMax) {

        gamma_ <- intern_expectation(fv_, K, theta_, gamma_);
        theta_ <- intern_maximisation(fv_, gamma_, theta_, K);

    }



    return(list("gamma"=gamma_,"theta"=theta_));
}


# read docs

data <- read.data(file.name = 'Task3A.txt', sample.size = 0, seed = 100, pre.proc = TRUE, spr.ratio = .99)

counts <- data$word.doc.mat

res_ <- dClust_eMax(counts)

label.hat <- apply(res_$gamma, 1, which.max)