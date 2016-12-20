rm(list=ls())
ls()

library(readr)
library(data.table)
train<- as.data.frame( fread( "fileName.csv" ))

train$Work_accident <- factor(train$Work_accident,levels = c(0,1),labels = c("No","Yes"))
train$left <- factor(train$left,levels = c(0,1),labels = c("No","Yes"))
train$promotion_last_5years <- factor(train$promotion_last_5years,levels = c(0,1),labels = c("No","Yes"))
train$salary <- factor(train$salary, levels = c("low","medium","high"), labels = c(1,2,3) )
train$sales <-   as.factor(train$sales)

# drop theses columns
col2DropinTrain <- c("last_evaluation","number_project","Work_accident","promotion_last_5years") # visually selected these

# Select attributes in train that are not in col2Drop
col2selTrain <- !names(train) %in% col2DropinTrain


# data set to use
train <- train[,col2selTrain]

library("caret")
library(randomForest)
set.seed(1000)

# response is a categorical
inTrain = createDataPartition(train$left, p=0.7, list = F)

training = train[ inTrain,]
testing = train[-inTrain,]

#variables
myNtree = 200
 #mtry <- sqrt(ncol(training))
myMtry = 3
myImportance = F

set.seed(825)

rfFit <- randomForest(training[,c(1:5)],training[,c("left")] ,  ntree=myNtree, mtry=myMtry, importance=myImportance)


rfPred <- predict(rfFit, testing)

table(rfPred,testing$left)
#         Reference
# Prediction   No  Yes
#         No  3428    0
#         Yes    0 1071

confusionMatrix(rfPred, rfPred)

out_of_sample_error <- sum(rfPred != testing$left)/length(testing$left)

print(out_of_sample_error)
# 0.0%

########### Plotting
#  plot function to plot the mean square error of the forest object
plot(rfFit)

#varImpPlot function to obtain the plot of variable importance
varImpPlot(rfFit)

## AUC CURVE
library(ROCR)

pred.rocr = prediction(as.numeric(rfPred),testing$left)

#perf.rocr = performance(pred.rocr, measure = "auc", x.measure ="cutoff")

perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")

plot(perf.tpr.rocr, colorize=T,main=paste("AUC:",(perf.rocr@y.values)))

#"NormalizedGini" is the other half of the metric. 
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) 
    accum.losses <- temp.df$actual / total.losses 
    gini.sum <- cumsum(accum.losses - null.losses) 
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}


NormalizedGini <- normalizedGini(as.numeric(rfPred),as.numeric(testing$left))

print(NormalizedGini)
# 1 ie 100% accuracy