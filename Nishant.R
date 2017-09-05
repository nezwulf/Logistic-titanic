install.packages("titanic")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("DAAG")
install.packages("caret")
library(titanic)
library(rpart.plot)
library(gmodels)
library(Hmisc)
library(pROC)
library(ResourceSelection)
library(car)
library(caret)
library(plyr)
library(dplyr)
library(InformationValue)
library(rpart)
library(randomForest)
library("DAAG")

cat("\014") # Clearing the screen

getwd()
setwd("C://YYYYYY//kfold") #This working directory is the folder where all the bank data is stored

# Train data
titanic_data_raw<-read.csv('train.csv')
titanic_data_fil <- subset(titanic_data_raw, select = c(2,3,5:8,10))
titanic_data_fil$Survived=as.factor(titanic_data_fil$Survived)
titanic_data_fil$Pclass=as.factor(titanic_data_fil$Pclass)

# replacing NA in Age with mean(age)

titanic_data_fil$Age[is.na(titanic_data_fil$Age)] <- mean(titanic_data_fil$Age, na.rm = T)
summary(titanic_data_fil)

set.seed(8888)

# Manual partitioning for GLM

Train <- createDataPartition(titanic_data_fil$Survived, p=0.7, list=FALSE)
titanic_training <- titanic_data_fil[ Train, ]
titanic_testing <- titanic_data_fil[ -Train, ]

titanic_model <- train(Survived ~ Pclass + Sex + Age + SibSp + 
                   Parch + Fare,  data=titanic_training, method="glm", family="binomial")

predictors(titanic_model)
exp(coef(titanic_model$finalModel))

predict(titanic_model)
predict(titanic_model, newdata=titanic_testing)
predict(titanic_model, newdata=titanic_testing, type="prob")

titanic_model

varImp(titanic_model)

titanic_model_1 <- train(Survived ~ Pclass + Sex + Age + SibSp +
                           Fare,  data=titanic_training, method="glm", family="binomial")
titanic_model_1

pred = predict(titanic_model, newdata=titanic_testing)
accuracy <- table(pred, titanic_testing[,"Survived"])
sum(diag(accuracy))/sum(accuracy)

pred = predict(titanic_model, newdata=titanic_testing)
confusionMatrix(data=pred, titanic_testing$Survived)

#K-fold cross validation built into GLM

ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

titanic_model_2 <- train(Survived ~ Pclass + Sex + Age + SibSp + 
                           Parch + Fare, data=titanic_data_fil, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

pred_1 = predict(titanic_model_2)
confusionMatrix(data=pred_1, titanic_data_fil$Survived)

accuracy_1 <- table(pred_1, titanic_data_fil[,"Survived"])
sum(diag(accuracy_1))/sum(accuracy_1)
