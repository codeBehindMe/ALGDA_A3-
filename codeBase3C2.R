####################### GENERAL AUXILIARY FUNCTIONS #######################
## The following structure helps us to have functions with multiple outputs
### credit: https://stat.ethz.ch/pipermail/r-help/2004-June/053343.html

error.rate <- function(Y1, T1){
  if (nrow(Y1)!=nrow(T1)){
    stop('error.rate: size of true lables and predicted labels mismatch')
  }
  return (sum(T1!=Y1)/nrow(T1))
}

##########################
options(warn=-1)
library(h2o)
#If there is a proxy: proxy.old <- Sys.getenv('http_proxy'); Sys.setenv('http_proxy'='');
localH2O =  h2o.init(nthreads = -1, max_mem_size = '2G', startH2O = TRUE)


# Students: Use the "absolute" path to the datasets on your machine (important)
labeled.frame <- h2o.importFile(path = 'C:/Users/aaron/OneDrive/Documents/Monash Data Science/Algorithms For Data Analysis/Assignment 1/assessments_datasets/Task3C_labeled.csv' ,sep=',') 
unlabeled.frame <- h2o.importFile(path = 'C:/Users/aaron/OneDrive/Documents/Monash Data Science/Algorithms For Data Analysis/Assignment 1/assessments_datasets/Task3C_unlabeled.csv' ,sep=',') 
test.frame <- h2o.importFile(path = 'C:/Users/aaron/OneDrive/Documents/Monash Data Science/Algorithms For Data Analysis/Assignment 1/assessments_datasets/Task3C_test.csv' ,sep=',') 

labeled.frame[,1] <- as.factor(labeled.frame$label)
unlabeled.frame[,1] <- NA
train.frame <- h2o.rbind(labeled.frame[,-1], unlabeled.frame[,-1])
test.frame[,1] <- as.factor(test.frame$label)

#
# build a neural network classifier based on the labeled training data
NN.model <- h2o.deeplearning(    
  x = 2:ncol(labeled.frame), # select all pixels + extra features
  y = 1,
  training_frame = labeled.frame, # specify the frame (imported file)    
  hidden = c(100), # number of layers and their units
  epochs = 50, # maximum number of epoches  
  activation = 'Tanh', # activation function 
  autoencoder = FALSE, # is it an autoencoder? Yes!
  l2 = 0.1
)

labeled.predict <- h2o.predict(NN.model, labeled.frame)$predict
error.rate(labeled.frame$label, labeled.predict)

# IV
test.predict <- h2o.predict(NN.model, test.frame)$predict
error.rate(test.frame$label, test.predict)


# II & III
reconstruction.train.error <- matrix(NA, nrow=20, ncol=1)
classification.labeled.error <- matrix(NA, nrow=20, ncol=1)

reconstruction.test.error <- matrix(NA, nrow=20, ncol=1)
classification.test.error <- matrix(NA, nrow=20, ncol=1)

## II & III
for (k in seq(20,500,20)) {
    # Students: need to write up code here
    NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(train.frame), # select all pixels + extra features
    training_frame = train.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)

    reconstruction.train.error[k / 20] <- mean(h2o.anomaly(NN.AutoEncoder, train.frame, per_feature = FALSE));

    #V A
    # get deepfeature and join with test.frame
    dfMatrixOut_ <- h2o.deepfeatures(NN.AutoEncoder, train.frame, layer = 1)
    allFeature_ <- h2o.cbind(train.frame, dfMatrixOut_)

    NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(test.frame), # select all pixels + extra features
    training_frame = test.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)

    reconstruction.test.error[k/20] <- mean(h2o.anomaly(NN.AutoEncoder, test.frame, per_feature = FALSE));
    print(paste("Finished k= ", k, " iter= ", k / 20))

    

    break;
}

stop("halt!")
# Produce the needed plots.
err_ <- as.data.frame(cbind(seq(20,500,20),reconstruction.test.error, reconstruction.train.error))
colnames(err_) <- c("Nurons","Test.Error", "Train.Error");

require(ggplot2)
require(reshape2)

err_ <- melt(err_, id = "Nurons")

ggplot(data = err_, aes(x = err_$Nurons, y = err_$value)) + geom_line(aes(color = as.factor(err_$variable)))


# V
# A.

NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(allFeature_), # select all pixels + extra features
    training_frame = train.frame, # specify the frame (imported file)    
    hidden = c(20), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)


dfMatrixOut_ <- h2o.deepfeatures(NN.AutoEncoder, train.frame, layer = 1)
allFeature_ <- h2o.cbind(train.frame, dfMatrixOut_)

labeled.predict <- h2o.predict(NN.model, test.frame)$predict
error.rate(labeled.frame$label, labeled.predict)



for (k in seq(20, 500, 20)) {
    # Autoencoder for the training set
    NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(labeled.frame), # select all pixels + extra features
    training_frame = labeled.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)

    # get the deep features.

    deepFeatures_ <- h2o.deepfeatures(NN.AutoEncoder,labeled.frame[,-1],layer = 1)



    # Announce loop progress.
    print(paste("Finished k= ", k, " iter= ", k / 20))

}

NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(labeled.frame), # select all pixels + extra features
    training_frame = labeled.frame, # specify the frame (imported file)    
    hidden = c(20), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)

# get the deep features.

deepFeatures_ <- h2o.deepfeatures(NN.AutoEncoder, labeled.frame[, -1], layer = 1)

allFeature_ <- h2o.cbind(labeled.frame, deepFeatures_)

NN.model <- h2o.deeplearning(
  x = 2:ncol(allFeature_), # select all pixels + extra features
  y = 1,
  training_frame = allFeature_, # specify the frame (imported file)    
  hidden = c(100), # number of layers and their units
  epochs = 50, # maximum number of epoches  
  activation = 'Tanh', # activation function 
  autoencoder = FALSE, # is it an autoencoder? Yes!
  l2 = 0.1)

labeled.predict <- h2o.predict(NN.model, allFeature_)$predict
error.rate(labeled.frame$label, labeled.predict)



augmentETrace_ <- matrix(NA, nrow = 20, ncol = 1)
for (k in seq(20, 500, 20)) {
    # Autoencoder for the training set
    NN.AutoEncoder <- h2o.deeplearning(
    x = 2:ncol(train.frame), # select all pixels + extra features
    training_frame = train.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE, # is it an autoencoder? Yes!
    l2 = 0.1)

    # get the deep features
    deepFeatures_ <- h2o.deepfeatures(NN.AutoEncoder, labeled.frame[, -1], layer = 1)
    # bind it to the labelled frame
    allFeature_ <- h2o.cbind(labeled.frame, deepFeatures_)

    # use the all feature to relearn nn classifer.
    NN.model <- h2o.deeplearning(
      x = 2:ncol(allFeature_), # select all pixels + extra features
      y = 1,
      training_frame = allFeature_, # specify the frame (imported file)    
      hidden = c(100), # number of layers and their units
      epochs = 50, # maximum number of epoches  
      activation = 'Tanh', # activation function 
      autoencoder = FALSE, # is it an autoencoder? Yes!
      l2 = 0.1)

    # predict
    labeled.predict <- h2o.predict(NN.model, allFeature_)$predict

    augmentETrace_[k / 20] <- error.rate(labeled.frame$label, labeled.predict)
    # Announce loop progress.
    print(paste("Finished k= ", k, " iter= ", k / 20))

}


plotdata_ <- as.data.frame(cbind(seq(20, 500, 20), augmentETrace_))

colnames(plotdata_) <- c("nFeatures","Error")
require(ggplot2)
ggplot(data=plotdata_,aes(x=plotdata_$nFeatures,y=plotdata_$Error)) +geom_line()