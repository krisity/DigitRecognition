#Some ideas: 
#1. https://vimeo.com/127790185 - video showing a GUI that classifies digits (written in C++ & OpenCV) -> no code available
#
#2. Learning from labelled data is what is called ?supervised learning?. It?s supervised because we?re taking 
#   the computer by hand through the whole training data set and ?teaching? it how the data that is linked with
#   different labels ?looks? like.
#
# Source for below: https://www.endpoint.com/blog/2017/05/30/recognizing-handwritten-digits-quick
#3. For the task of classifying I decided to use the XGBoost library which is somewhat a hot new technology in 
#   the world of machine learning. It?s an improvement over the so-called Random Forest algorithm. The reader 
#   can read more about XGBoost on its website: https://xgboost.readthedocs.io/.

#Get the current working directory:
getwd()

#Change the working directory to more appropriate one
setwd("D:/TU-Project/Digit Recognizer")


#Load the data sets - test and train which were first downloaded from kaggle

train <- read.csv("train.csv", header=TRUE)

test <- read.csv("test.csv", header=TRUE)

####################################################
#Retain only 10% of the data to make things faster.
smallTrain <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]
####################################################

#Extract the column with thelabels from the train dataset
trainy <- train[,1] 

#Extract the data without the labels
trainx <- train[,-1]

#Load the library that we are going to use to fit the random forest method

library(randomForest)
library(nnet) #neural networks
library(e1071) #used for Naive Bayes approach &  Support Vector Machine (SVM)
library(Rtsne) #for t-sne
library(xgboost) #something similar to RF, not yet used here
library(RColorBrewer) #used for the barplot
library(tidyverse)
library(pacman)


################################################################################
##### SOME ILLUSTARTIVE FUNCTIONS FOR BETTER VISUALIZATION OF THE DATASETS #####
################################################################################

#Shows the 1st row of the train dataset
train[1:1,]

#shows the 1st column  of the train dataset which holds the labels of the numbers
train[,1:1]

#For testing purposes, create a smaller training set containing the forst 10000 rows 
#of the original training set
small_train <- train[c(1:10000),]

#Get dimensions of the dataset - as it says on kaggle, 1st column should reflect 
#the labels and the remaining 28x28 = 784 columns should be the pixels with 
#different values (0-255) of each digit.
dim(train)

#TO BE ADDED - There should be a summary table with min/max values to show that the intensity of the
#pixels in each image are between 0 & 255 as it should be (0 = black, 255 = white, everything in-between 
#is some shade of gray)
summary(train)

#TO BE TESTED - Data cleaning -> Are there any NA values, duplicated values etc

# Duplicated rows
sum(duplicated(test)) # no duplicate rows
sum(duplicated(train)) # no duplicate rows

# Checking for NAs
sum(sapply(test, function(x) sum(is.na(x)))) # There are no missing values
sum(sapply(train, function(x) sum(is.na(x)))) # There are no missing values

#Get all digit labels from the train dataset and convert them to a factor variable
labels <- as.factor(train[,1])

#Summary in form of a table showing how many times a digit appears in the train dataset (same
#as the histogram but in the form of a table)
summary(labels)

#Show the detailed structure of the train dataset
str(train)

#Get the type of each column - in this case all columns should be of type
#integer holding integer values
sapply(train, typeof)

#Shows a histogram so that we can see how many times a digit is 
#presented in the whole dataset
barplot(table(labels), main="Total Number of Digits (Training Set)", col=brewer.pal(10,"Set3"),
        xlab="Numbers", ylab = "Frequency of Numbers")

#Try and represent a single row into a matrix to display similar image as this one - 
#https://blog.qarnot.com/wp-content/uploads/2017/02/encoding.png

testRow <- train[9,]
testRow <- testRow[,-1]
testRowMatrix <- matrix(testRow,28,28,byrow=T) #fill in by rows
testRowMatrix #check to see what is the digit

#USEFUL - This bit of code displays the first 36 digits visually with the label next to
#them

trainasmatrix <- as.matrix(train)

plotTrain <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(trainasmatrix[i, -1], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, train[i,1])
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTrain(1:36)

#This bit of code displays an image - done as a function
displayDigit <- function(X){ 
  
  m <- matrix(unlist(X),nrow = 28,byrow = T) 
  
  m <- t(apply(m, 2, rev)) 
  
  image(m,col=grey.colors(255))}

#Display the digit from the 11th row in black and white - in thhis case the
#digit is 8.
displayDigit(train[11,-1])


#We can quickly plot the pixel color values to obtain a picture of the digit.

#Create a 28*28 matrix with pixel color values (byrow means that the row is organized
#by 28 chunks of row)
m = matrix(unlist(train[11,-1]),nrow = 28,byrow = T)

# Plot that matrix
rotate <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)


# Plot the pixel values of the selected digit  - in this case the same digit from 
# the same row (row 11, digit is 8)
m

# Plot a bunch of black and white images
par(mfrow=c(2,3))

lapply(1:6, 
       function(x) image(
         
         rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         
         col=grey.colors(255),
         
         xlab=train[x,1]
         
       )
)

#Set plot options back to default
par(mfrow=c(1,1)) 


#This bit of code takes some time, but visualizes digits from 0 to 9 written
#in different  ways. It can be seen that 7 and 1 are visually very similar

for (i in seq(0, 9)) {
  sample <- train[train$label == i, ]
  # Omit label column 
  sample <-  sample[ ,-1]
  # Resetting the margins
  par(mar = c(1,1,1,1))  
  # Build 10 rows by 5 column Plot matrix  for each digit 
  par(mfrow = c(4,10)) 
  # 50 samples between 10 & 5000 
  for (j in seq(10, 4000, by = 100)) {
    # Build a 28 X 28 matrix of pixel values in each row
    digit <- t(matrix(as.numeric(sample[j, ]), nrow = 28)) 
    # Inverse the pixel matrix to get the image of the number right
    image(t(apply(digit, 2, rev)), col = grey.colors(255))
  }
}

################################################################################
#################### VISUALIZING TRAIN DATASET USING t-SNE #####################
################################################################################

# Source: https://rpubs.com/dhnanjay/236153 
# Source #2: https://lvdmaaten.github.io/tsne/

#t-SNE stands for t-Distributed Stochastic Neighbor Embedding. It visualizes 
#high-dimensional data by giving each data point a location in a two or 
#three-dimensional map. It is a variation of Stochastic Neighbor Embedding 
#that allows optimization, and produces significantly better visualizations
#by reducing the tendency to lump points together in the center of the map
#that often renders the visualization ineffective and unreadable. 
#Since it visualizes high-dimensional data, it is one of the best technique to 
#visualize a large data set such as MNIST.

#Load the RSNE library
library(Rtsne)

#Visualize using t-SNE 
#tsne <- Rtsne(as.matrix(train), check_duplicates = TRUE, pca = TRUE, perplexity = 30, theta = 0.5, dims = 2)

train <- read.csv("train.csv", header=TRUE)

#Get 80% of the training data
small.train <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Save only the column with the labels of the training dataset
small.train.label.only <- small.train[,1]

#Remove the "label" column from the train dataset
small.train.without.labels <- small.train[,-1] 

#Strart the timer
start.time <- proc.time()

#Apply t-sne on the small training dataset
tsne <- Rtsne(small.train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

#Calculate elapsed time
proc.time() - start.time

#for 33600 observations
#   user  system elapsed 
#637.29    5.62  681.38 

#Set colors of the plot
colors = rainbow(length(unique(small.train$label)))

#Set additional settins of the colors on the plot to identify the labels
names(colors) = unique(small.train$label)

#Plot the results of the tsne
plot(tsne$Y, t='n', main="tsne")

#Add text to the plot
text(tsne$Y, labels=small.train$label, col=colors[small.train$label])

#Plot the clusters in more understandable way
library(ggplot2)

tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], col = as.factor(small.train.label.only))

ggplot(tsne_plot) + ggtitle("t-SNE on the training dataset") + geom_point(aes(x=x, y=y, color=col))

#Confusion matrix
table(`Actual Class` = small.train.label.only, `Predicted Class` = tsne_plot$col)

#overall prediction accuracy of our model - first calculate the error rate from the knn
error.rate.tsne <- sum(small.train.label.only != tsne_plot$col)/nrow(small.train)

#Accuracy
accuracy <- round((1 - error.rate.tsne) *100,2)

#Accuracy is 100% which is misleading. 
accuracy

tsne$Y

xy <- as.data.frame(tsne)

colnames(xy) <- c('x', 'y')

xy$label <- small.train$label

head(xy)

# Source #2: https://www.kaggle.com/yidhir/visualisation

################################################################################
########################### KNN ON THE TRAINING DATASET ########################
################################################################################

#

### NB! Here - http://www.ryanzhang.info/analytic/kaggle-digit-recognizer-revisited-using-r/ 
### is said that KNN for the test data with 28000 rows takes nearly 3 hours to complete!!!!!!

library(RWeka) # needed for the IBk function below

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

train <- read.csv("train.csv", header=TRUE)

#Take 80% of the original training data to see how long it would take to 
#fit the model
small.train <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Extract 10% of the otiginal training data which will act as test data:
test.data <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Create a factor variable which will hold only the label variables from the test
#dataset
test.data.label.only <- test.data[,1]

#Remove the "label" column from the test dataset
test.data <- test.data[,-1] 

#Start calculating how much time the implementation of the model would take
pc <- proc.time()

#Fit the model on small train data ~ 33600 observations instead of 42000
model.knn <- IBk(small.train$label ~ ., data = small.train)

#See how much time elapsed since the model was fitted
#NB! it takes a lot of time
proc.time() - pc

#For 4200 variables it took:
#   user   system  elapsed 
#  138.19   0.36    150.56 

#For 33600 variable it took:
#   user  system elapsed 
#  7062.73    1.97 7191.50


#The confusion Matrix will give us the idea of how precise and accurate our
#Model was for each digit. 
prediction.knn <- predict(model.knn, newdata = test.data, type = "class")

#Table showing the actual vs predicted classes
table(`Actual Class` = test.data.label.only, `Predicted Class` = prediction.knn)

#overall prediction accuracy of our model - first calculate the error rate from the knn
error.rate.knn <- sum(test.data.label.only != prediction.knn)/nrow(test.data)

#Accuracy
accuracy <- round((1 - error.rate.knn) *100,2)

#Accuracy is pretty good - 99.43% (for 33600 observations & 4200 test observations)
accuracy

#Print 1 line which has the accuracy in %
print(paste0("Prediction Accuracy: ", accuracy))

#Predict Digit for kNN
row <- 1

#Predict a digit from test dataset
prediction.digit <- as.vector(predict(model.knn, newdata = test.data[row,  ], type = "class"))

#Print the current digit  (digit on the 1st row)
print(paste0("Current Digit: ", as.character(test.data.label.only[row])))

#Print the predicted digit as per the KNN algorithm
print(paste0("Predicted Digit: ", prediction.digit))

#Create a matrix that will hold the test dataset. The matrix would be used
#for the funciton below:
testasmatrix <- as.matrix(test.data)

#Function that will plot digits from the test dataset. On the left side of each 
#image in the plot, the predicted digit will be shown in white. On the right side
#in orange it will be shown the actual digit only if the model did not classify the
#digit correctly.
plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, prediction.knn[i])
    predicted.digit <- prediction.knn[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)

library(modelr)
library(caret)

#Root Mean Square Error (RMSE) is the standard deviation of the residuals
#(prediction errors). Residuals are a measure of how far from the regression
#line data points are; RMSE is a measure of how spread out these residuals
#are. In other words, it tells you how concentrated the data is around the 
#line of best fit. 
#Lower values of RMSE indicate better fit.
rmse <- RMSE(prediction.knn, test.data.label.only)
rmse

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - prediction.knn)^2)
mse


################################################################################
#################### FAST NEAREST NEIGHBORS (FNN) ALGORITHM ####################
############################# ON THE TRAINING DATASET ##########################
################################################################################

#Source: http://rstudio-pubs-static.s3.amazonaws.com/6287_c079c40df6864b34808fa7ecb71d0f36.html

library(FNN)  # Fast k-Nearest Neighbors (kNN)
library(e1071)  # Support Vector Machine (SVM)

#Again we are using small train dataset and small test dataset for testing purposes

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

small.train <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Extract 10% of the otiginal training data which will act as test data:
test.data <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Create a factor variable which will hold only the label variables from the test
#dataset
test.data.label.only <- test.data[,1]

#Remove the "label" column from the test dataset
test.data <- test.data[,-1] 

#start time
pc <- proc.time()

# Avoid Name Collision (knn) and fit the model
model.fnn <- FNN::knn(small.train[, -1], test.data, small.train$label, 
                      k = 10, algorithm = "cover_tree")

#Calculate how much it took for the model to be fitted
proc.time() - pc

#For 4200 observations it took
#   user   system   elapsed 
#  92.27    0.37   112.45 

#For 33600 observations it took
#   user  system elapsed 
# 749.20    1.28  813.65

#summary of the model
summary(model.fnn)

model.fnn

#Print the confusion matrix
table(`Actual Class` = test.data.label.only, `Predicted Class` = model.fnn)

#Print the error rate
error.rate.fnn <- sum(test.data.label.only != model.fnn)/nrow(test.data)

print(paste0("Accuary (Precision): ", 1 - error.rate.fnn))

#Save the fnn predictions as numeric values
model.fnn1  <- as.numeric(model.fnn1)

#Lower values of RMSE indicate better fit.
rmse <- RMSE(model.fnn1, test.data.label.only)
rmse # 0.82

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - model.fnn1)^2)
mse #0.67


#Prediction for the 1st row/digit
row <- 1
prediction.digit <- model.fnn[row]

print(paste0("Current Digit: ", as.character(test.data.label.only[row])))

print(paste0("Predicted Digit: ", prediction.digit))

testasmatrix <- as.matrix(test.data)

#Same function as the one in EDA to see the classified digits with the fnn model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, model.fnn[i])
    predicted.digit <- model.fnn[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)


################################################################################
################## NAIVE BAYES ALGORITHM ON THE TRAINING DATASET ###############
################################################################################

#Source: http://rstudio-pubs-static.s3.amazonaws.com/6287_c079c40df6864b34808fa7ecb71d0f36.html

#NB! When fitting the model, the "label" column must be converted
#to factor variable in order for the model to work out.
#Y values in naiveBayes function should be categorical (in our case 
#they are numeric). This is why we convert them to factor

#Again using the training dataset which is divided into 2: test and train
train <- read.csv("train.csv", header=TRUE)

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

#Get  80% from the training data which would server as train data
small.train.with.labels <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

test.data.label.only <- test.data.with.labels[,1]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 

library(e1071)

pc <- proc.time()

#Fit the model on 33600 observations from the training dataset
model.naiveBayes <- naiveBayes((as.factor(small.train.with.labels$label)) ~ ., data = small.train.with.labels)

proc.time() - pc

#Fitting the model took fairly small amount of time - for 4200 obse3rvations:
#   user  system elapsed 
#   4.11    0.39   21.04 

#Fitting the model to 33600 observations
#   user  system elapsed 
#   6.33    1.04   94.97 


#summary of the model
summary(model.naiveBayes)

#Predict digits from the test data
#NB! - it takes some time
prediction.naiveBayes <- predict(model.naiveBayes, newdata = test.data.with.labels, type = "class")

prediction.naiveBayes

#Confusion matrix with actual and predicted digits
table(`Actual Class` = test.data.label.only, `Predicted Class` = prediction.naiveBayes)

#Error rate
error.rate.naiveBayes <- sum(test.data.label.only != prediction.naiveBayes)/nrow(test.data.without.labels)

#Print the accuracy - 0.5197
print(paste0("Accuary (Precision): ", 1 - error.rate.naiveBayes))

library(caret)

rmse <- RMSE(as.integer(prediction.naiveBayes), test.data.label.only)
rmse #3.34

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
prediction.naiveBayesIntege <- as.integer(prediction.naiveBayes)
mse <- mean((test.data.label.only - prediction.naiveBayesIntege)^2)
mse #11.13571

# Prediction for Row 1
row <- 1

#Extract the predicted digit
prediction.digit <- as.vector(predict(model.naiveBayes, newdata = test.data.without.labels[row, 
                                                                               ], type = "class"))
print(paste0("Current Digit: ", as.character(test.data.label.only[row])))

print(paste0("Predicted Digit: ", prediction.digit))


testasmatrix <- as.matrix(test.data.without.labels)

#Same function as the one in EDA to see the classified digits with the naive
#bayes model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, prediction.naiveBayes[i])
    predicted.digit <- prediction.naiveBayes[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)


################################################################################
##################### SVM (SUPPORT VECTOR MACHINE) ALGORITHM ###################
############################# ON THE TRAINING DATASET ##########################
################################################################################

#Source: https://rpubs.com/Adityanakate/240186

#NB!!!!!!! Model works great (~94% accuracy), but when we plot the actual vs 
# predicted digits, there is a huge difference....

#2nd try is better (algorithm shown below this one)

#Again using the training dataset which is divided into 2: test and train

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

library(foreach)

#Get  10% from the training data which would server as train data
small.train.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

test.data.label.only <- test.data.with.labels[,1]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 

small.train.with.labels$label <- as.factor(small.train.with.labels$label)


set.seed(0)

# Load the libs required for the analysis
library(class)
library(readr)
library(caret)
library(e1071)

train <- small.train.with.labels
test <- test.data.without.labels


#splitting train data in TRAIN and TEST again
rows <- sample(1:nrow(train), 3000) 
labels <- as.factor(train[rows,1])
train_train <- train[rows,-1]


#Applying PCA
pca.train <- prcomp(train_train, scale=FALSE, center = T)

varEx<-as.data.frame(pca.train$sdev^2/sum(pca.train$sdev^2))
varEx<-cbind(c(1:784),cumsum(varEx[,1]))
colnames(varEx)<-c("Nmbr_PCs","Cum_Var")
VarianceExplanation<-varEx[seq(0,700,50),]

rotate<-pca.train$rotation[,1:50]
trainFinal2<-as.matrix(scale(train_train,center = TRUE, scale=FALSE))%*%(rotate) 

# SVM models
pc <- proc.time()
svm.fit <- svm(trainFinal2,labels, kernel='radial') 
proc.time() - pc

summary(svm.fit) 

#Checking on train data
yhat <- predict(svm.fit,trainFinal2 )
confusionMatrix(yhat, train[rows,1])

#Checking on test data
trainMeans<-colMeans(train_train)
trainMeansMatrix<-do.call("rbind",replicate(nrow(train[-rows,]),trainMeans,simplif=FALSE))
testFinal<-as.matrix(train[-rows,-1]-trainMeansMatrix) 
testfinal2<-as.matrix(testFinal)%*%(rotate) 

yhat <- predict(svm.fit,testfinal2 )
confusionMatrix(yhat, train[-rows,1])

testasmatrix <- as.matrix(test)

#Same function as the one in EDA to see the classified digits with the svm model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, yhat[i])
    predicted.digit <- yhat[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)


################################################################################
##################### SVM (SUPPORT VECTOR MACHINE) ALGORITHM ###################
############################# ON THE TRAINING DATASET ##########################
################################################################################

#Source: https://www.kaggle.com/yidhir/testing-svm

#Another try for SVM - SIGNIFICANTLY BETTER + MORE UNDERSTANDABLE 

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train_original"))

train_original <- read.csv("train.csv", header=TRUE)

#Get  80% from the training data which would serve as train data
small.train.with.labels <- train_original[sample(1:nrow(train_original), 0.8*nrow(train_original), replace = FALSE), ]

#All labels from the train dataset - stored as factor
train.data.label.only <- small.train.with.labels[,1]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train_original[sample(1:nrow(train_original), 0.1*nrow(train_original), replace = FALSE), ]

#Factor variable holding all labels of the test dataset
test.data.label.only <- test.data.with.labels[,1]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 

library(readr)

library(e1071)

#removing the label column from the train dataset
small.train.without.labels = small.train.with.labels[-1]

#Start the timer
pc <- proc.time()

#Fitting the model
model = svm(small.train.without.labels,y=as.factor(train.data.label.only), kernel = "polynomial",degree = 2)

#End time
proc.time() - pc

#   user  system elapsed 
#872.14    2.57  921.11 

#Get model summary statistics
summary(model)

#Predict the labels on the test dataset - the one without labels
pred = predict(model, test.data.without.labels)

#Calculate the error
error.rate.svm <- sum(test.data.label.only != pred)/nrow(test.data.without.labels)

#Print the accuracy in the console
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))

#Save all predictions as a data frame object with 2 columns: "ImageId" & "Label"
prediction = data.frame(ImageId=1:nrow(test.data.without.labels), Label=pred)

#Create and print a table / confusion matrix showing actual vs predited digits
table(`Actual Class` = test.data.label.only, `Predicted Class` = pred)

#Lower values of RMSE indicate better fit.
rmse <- RMSE(as.integer(pred), test.data.label.only)
rmse # 1.051529

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - as.numeric(pred))^2)
mse #1.105714


#Create a matrix variable that hold the test dataset
testasmatrix <- as.matrix(test.data.without.labels)

#Same function as the one in EDA to see the classified digits with the svm model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, pred[i])
    predicted.digit <- pred[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)

################################################################################
##################### KSVM (SUPPORT VECTOR MACHINE) ALGORITHM ##################
############################# ON THE TRAINING DATASET ##########################
################################################################################

#Source: https://www.kaggle.com/lucianolattes/classifying-with-svm-using-polydot-ker

#NB!!!!! - using ksvm works (https://cran.r-project.org/web/packages/kernlab/kernlab.pdf)

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train_original"))

#Train dataset with 42000 rows
train_original <- read.csv("train.csv", header=TRUE)

#Get  80% from the training data which would server as train data
small.train.with.labels <- train_original[sample(1:nrow(train_original), 0.8*nrow(train_original), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train_original[sample(1:nrow(train_original), 0.1*nrow(train_original), replace = FALSE), ]

test.data.label.only <- test.data.with.labels[,1]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 


library(readr)
library(kernlab)

train <- small.train.with.labels

#Convert the label column to factor instead of integer
train$label <- as.factor(train$label)

#don't know what this is for
numTrain <- 4000

#set.seed function in R is used to reproduce results i.e. it produces the same
#sample again and again. When we generate randoms numbers without set.seed() 
#function it will produce different samples at different time of execution.
set.seed(13)

#Sample the first 4000 rows from the small train dataset
rows <- sample(1:nrow(train), numTrain)

#Create train2 dataset which holds 4000 observations only
train2 <- train[rows, ]

#Start the timer
pc <- proc.time()

# ksvm belongs to kernlab library; using a degree 3 polydot we can obtain similar
#results than the basic random forest example.
#Fit the ksvm model
filter <- ksvm(label ~ ., data = train, kernel = "polydot", kpar = list(degree = 3), cross = 3)

#End time
proc.time() - pc

#Check the summary of the model/algorithm
filter

#Predict labels of the test dataset based on the ksvm model above
labels <- predict(filter, test.data.without.labels)

#Accuracy
sum(labels == test.data.with.labels$label) / nrow(test.data.with.labels)

#Convert the predictions to dataframe object  that would be suitable for 
#submission to kaggle
predictions <- data.frame(ImageId=1:nrow(test.data.without.labels), Label=levels(train$label)[labels])

#Create and print a table / confusion matrix showing actual vs predited digits
table(`Actual Class` = test.data.with.labels$label, `Predicted Class` = labels)

library(caret)

#Lower values of RMSE indicate better fit.
rmse <- RMSE(as.integer(labels), test.data.label.only)
rmse # 1.050737

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - as.numeric(labels))^2)
mse #1.104048


#Convert the test dataset to matrix in order to be used in the below function
testasmatrix <- as.matrix(test.data.without.labels)

#Same function as the one in EDA to see the classified digits with the ksvm model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, labels[i])
    predicted.digit <- labels[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)



################################################################################
############# RPart (Recursive Partitioning and Regression Trees) ##############
###################### ALGORITHM ON THE TRAINING DATASET #######################
################################################################################

#Source: http://rstudio-pubs-static.s3.amazonaws.com/6287_c079c40df6864b34808fa7ecb71d0f36.html

#We are testing the algorithm on the training set

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

#Load the training data set
train <- read.csv("train.csv", header=TRUE)

#Get 80% from the training data which would server as train data
small.train.with.labels <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 

#Labels from the test data set
test.data.label.only <- test.data.with.labels[,1]


#Start the timer
pc <- proc.time()

library(rpart)

#Fit the model to the training set
model.rpart <- rpart(small.train.with.labels$label ~ ., method = "class", data = small.train.with.labels)

#Calculate elapsed time
proc.time() - pc

#   user  system elapsed 
# 162.57    1.19  169.05 

#Print summary of the model
printcp(model.rpart)

#Plot model as tree
plot(model.rpart, uniform = TRUE, main = "Classification (RPART). Tree of Handwritten Digit Recognition ")

#Add text to the plot
text(model.rpart, all = TRUE, cex = 0.75)

library(maptree)

#Unexplainable tree for rpart....
draw.tree(model.rpart, cex = 0.5, nodeinfo = TRUE, col = gray(0:8/8))

#Predict on test data.
prediction.rpart <- predict(model.rpart, newdata = test.data.without.labels, type = "class")

#Visualize table with actual vs predicted classes based on RPart algorithm
table(`Actual Class` = test.data.with.labels$label, `Predicted Class` = prediction.rpart)

#Calculate and print the error rate/accuracy
error.rate.rpart <- sum(test.data.with.labels$label != prediction.rpart)/nrow(test.data.with.labels)

print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))

#Lower values of RMSE indicate better fit.
rmse <- RMSE(as.numeric(prediction.rpart), test.data.label.only)
rmse # 2.80

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - as.numeric(prediction.rpart))^2)
mse #7.854524


#Declare row 1
row <- 1

#Prediction for 1st row
prediction.digit <- as.vector(predict(model.rpart, newdata = test.data.with.labels[row, 
                                                                          ], type = "class"))

#Print the current digit in the console
print(paste0("Current Digit: ", as.character(test.data.with.labels$label[row])))

#Print the predicted digit in the console according to the RPart algorithm
print(paste0("Predicted Digit: ", prediction.digit))

testasmatrix <- as.matrix(test.data.without.labels)

#Same function as the one in EDA to see the classified digits with the rpart model.
#white digits = predicted digits by the model for the test data
#orange digits = actual digits from the test data (they appear only if there
#is a difference between actual and predicted digits)

plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, prediction.rpart[i])
    predicted.digit <- prediction.rpart[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)

################################################################################
###################### RANDOM FOREST ON THE TRAINING DATASET ###################
################################################################################

#Another RF model (https://github.com/GregHamel/Kaggle/blob/master/digit_recognizer/digit_recognizer_writeup.Rmd) 

#Source: https://rpubs.com/meisenbach/284590

#Load the training data set
train <- read.csv("train.csv", header=TRUE)

#Get 80% from the training data which would server as train data
small.train.with.labels <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Remove the "label" column from the test dataset
test.data.without.labels <- test.data.with.labels[,-1] 

#Labels from the test data set
test.data.label.only <- test.data.with.labels[,1]

#Labels from the test data set
train.data.label.only <- as.factor(small.train.with.labels[,1])

#Set number of trees
numTrees <- 25

# Train on entire training dataset and predict on the test
#Set start time
startTime <- proc.time()

library(randomForest)

#RF model - takes about 20 minutes
rf <- randomForest(small.train.with.labels[-1], train.data.label.only, xtest=test.data.without.labels, 
                   ntree=numTrees)

proc.time() - startTime

#Summary of RF model
rf

#output predictions for submission
predictions <- data.frame(ImageId=1:nrow(test.data.without.labels), 
                          Label=levels(train.data.label.only)[rf$test$predicted])

head(predictions)

#Calculate and print the error rate/accuracy
error.rate.rpart <- sum(test.data.label.only != rf$test$predicted)/nrow(test.data.with.labels)

print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))

#Lower values of RMSE indicate better fit.
rmse <- RMSE(as.numeric(predictions[,-1]), test.data.label.only)
rmse # 1.060548

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - as.numeric(predictions[,-1]))^2)
mse #1.124762

testasmatrix <- as.matrix(test.data.without.labels)

#Same function as the one in EDA to see the classified digits with the nnet model.
plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, predictions[i,2])
    predicted.digit <- predictions[i,2]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)


################################################################################
################## NEURAL NETWORK ON 90% OF THE TRAINING DATASET ###############
################################################################################

#Source: https://www.kaggle.com/somykamble/nnet-in-r

library(nnet)
library(caret)

#Remove all variables except for "train" variable
rm(list=setdiff(ls(), "train"))

train <- read.csv("train.csv", header=TRUE)

options(digit=3)

seed=1

#Get 80% from the training dataset which would serve as training data
digit.data <- train[sample(1:nrow(train), 0.8*nrow(train), replace = FALSE), ]

#Get 10% from the training dataset which would serve as test data
test.data.with.labels <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

test.data.label.only <- test.data.with.labels[,1]

#Remove the "label" column from the test dataset
test.data <- test.data.with.labels[,-1]

#Get the labels from the "label" column as a factor variable with 9 levels (digits 
# 1 to 9)
digit.data$label<- factor(digit.data$label,levels=0:9)

#Remove the "label" column from the train dataset
digit.x<-digit.data[,-1]

#Keep only the "label" column from the train dataset
digit.y<-digit.data[,1]

#Start the time
start.time <- proc.time()

#Fit the neural network model to the train dataset
digit.model<- train(x=digit.x,y=digit.y,
                    method="nnet",
                    tuneGrid=expand.grid(
                      .size=c(6),
                      .decay=0.1),
                    trControl=trainControl(method='none',seeds=seed),
                    MaxNWts=10000,
                    maxit=100)

#End time
proc.time() - start.time

#Predict new examples by a trained neural net.
digit.preds<- predict(digit.model)

#Print the confusion matrix to see the overall statistics including the accuracy
#of the model
confusionMatrix(xtabs(~digit.preds + digit.y))

#Use the neural network model for the test dataset
sam <- predict(digit.model,test.data)

plotnet(digit.model$finalModel, y_names = "income level")
plotnet(digit.model)

#Write the fail with the predicted values
write.csv(sam,file='submisssion.csv')

#Lower values of RMSE indicate better fit.
rmse <- RMSE(as.numeric(sam), test.data.label.only)
rmse # 2.363563

#The MSE is a measure of the quality of an estimator?it is always
#non-negative, and values closer to zero are better
mse <- mean((test.data.label.only - as.numeric(sam))^2)
mse #5.586429



testasmatrix <- as.matrix(test.data)

#Same function as the one in EDA to see the classified digits with the nnet model.
plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, sam[i])
    predicted.digit <- sam[i]
    actual.digit <- test.data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, test.data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)



################################################################################
###################### K-means ###################
################################################################################

#source: https://www.kaggle.com/satishtiwari23/clustering-by-using-k-means-predictive-analysis

rm(list=setdiff(ls(), "train"))

train <- read.csv("train.csv", header=TRUE)

#Get 10% of the training data
my_data <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Get the label column
my_data.label.only <- my_data[,1]

#my_data.label.only <- as.factor(my_data.label.only)

#GEt test data
test <- train[sample(1:nrow(train), 0.1*nrow(train), replace = FALSE), ]

#Remove the "label" column from the dataset
my_data <- my_data[,-1]

k <- 10

#myKmeans = kmeans(my_data, k, iter.max = 100,nstart=20)

#myKmeans <- kmeans(my_data, 10, iter.max = 50, nstart = 50)

#clusters <- myKmeans$cluster

#summary(as.factor(my_data.label.only))

#summary(as.factor(clusters))

#df<-data.frame(predicted = myKmeans$cluster, actual = my_data.label.only)


#tb<-table(pred=df$predicted,actual=df$actual)

######
#source: https://bradleyboehmke.github.io/HOML/kmeans.html

mnist_clustering <- kmeans(my_data, centers = 10, nstart = 10)

str(mnist_clustering)

# Extract cluster centers
mnist_centers <- mnist_clustering$centers

# Plot typical cluster digits
par(mfrow = c(2, 5), mar=c(0.5, 0.5, 0.5, 0.5))
layout(matrix(seq_len(nrow(mnist_centers)), 2, 5, byrow = FALSE))
#Fig. - Cluster centers for the 10 clusters identified in the MNIST training data.
for(i in seq_len(nrow(mnist_centers))) {
  image(matrix(mnist_centers[i, ], 28, 28)[, 28:1], 
        col = gray.colors(12), xaxt="n", yaxt="n")
}


# Create mode function
mode_fun <- function(x){  
  which.max(tabulate(x))
}

mnist_comparison <- data.frame(
  cluster = mnist_clustering$cluster,
  actual = my_data.label.only
) %>%
  group_by(cluster) %>%
  mutate(mode = mode_fun(actual)) %>%
  ungroup() %>%
  mutate_all(factor, levels = 0:9)

# Create confusion matrix and plot results
#Confusion matrix illustrating how the k-means algorithm clustered the digits (x-axis) and the actual labels (y-axis).
library(yardstick)
yardstick::conf_mat(
  mnist_comparison, 
  truth = actual, 
  estimate = mode
) %>%
  autoplot(type = 'heatmap')


tb<-table(pred=mnist_clustering$cluster,actual=my_data.label.only)

tb


testasmatrix <- as.matrix(my_data)


#Same function as the one in EDA to see the classified digits with the nnet model.
plotTest <- function (images){
  op <- par(no.readonly = TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images) { # reverse and traspose each matrix to rotate images
    m <- matrix(testasmatrix[i,], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col = grey.colors(255),axes=FALSE)
    text(0.05, 0.2, col="white", cex =1.2, mnist_clustering$cluster[i])
    predicted.digit <- mnist_clustering$cluster[i]
    actual.digit <- my_data.label.only[i]
    if (predicted.digit != actual.digit) {
      text(0.95, 0.2, col="darkorange", cex =1.2, my_data.label.only[i])
      
    }
  }
  
  par(mfrow=c(1,1))  # reset the original graphics parameters
}

plotTest(1:36)
