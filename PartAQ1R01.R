## Assignment3PartAQ1R01



utils_readData <- function(file.name = './assessments_datasets/Task3A.txt', sample.size = 1000, seed = 100, pre.proc = TRUE, spr.ratio = 0.90) {
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


utils_logSum <- function(v) {
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


utils_initialParams <- function(vocab_size, K = 4, seed = 123456) {

    # vocab size is the counte vectors number of rows. (the count vector has words as rows and document number as columns).

    rho <- matrix(1 / K, nrow = K, ncol = 1) # assume all clusters have the same size (we will update this later on)
    mu <- matrix(runif(K * vocab_size), nrow = K, ncol = vocab_size) # initiate Mu 
    mu <- prop.table(mu, margin = 1) # normalization to ensure that sum of each row is 1
    return(list("rho" = rho, "mu" = mu))
}


E_step <- function(gamma, model, counts,hard=FALSE) {
    # Model Parameter Setting
    N <- dim(counts)[2] # number of documents
    K <- dim(model$mu)[1]

    # E step:    
    for (n in 1:N) {
        for (k in 1:K) {
            ## calculate the posterior based on the estimated mu and rho in the "log space"
            gamma[n, k] <- log(model$rho[k, 1]) + sum(counts[, n] * log(model$mu[k,]))
        }
        # normalisation to sum to 1 in the log space
        logZ = logSum(gamma[n,])
        gamma[n,] = gamma[n,] - logZ
    }

    # converting back from the log space 
    gamma <- exp(gamma)



    return(gamma)
}

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


train_obj <- function(model, counts) {

    ##--- the training objective function --------------------------------------------
    # Input: 
    #    model:  the model object containing the mu and rho
    #    counts: the word-document frequency matrix
    # Output:
    #    nloglike: the negative log-likelihood i.e. log P(counts|model) 
   

    N <- dim(counts)[2] # number of documents
    K <- dim(model$mu)[1]

    nloglike = 0
    for (n in 1:N) {
        lprob <- matrix(0, ncol = 1, nrow = K)
        for (k in 1:K) {
            lprob[k, 1] = sum(counts[, n] * log(model$mu[k,]))
        }
        nloglike <- nloglike - logSum(lprob + log(model$rho))
    }

    return(nloglike)
}


# Main body


# read data
dts_ <- utils_readData();

# get the word-document count matrix
counts_ <- dts_$word.doc.mat

# get number of docs
nDocs_ <- dim(counts_)[2]
# get number of words
nWords_ <- dim(counts_)[1]


# number of clusters expected to find
K <- 4

# initialise parameters
mdl_ <- utils_initialParams(nWords_);
gamma_ <- matrix(0,nrow = nDocs_,ncol = K)