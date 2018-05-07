library(readr)
library(dplyr)
library(caret)


dataset <- read_csv("data/base_crabs.csv")
set.seed(1234)

index <- createDataPartition(dataset$class, p=0.70, list=FALSE)
trainData = dataset[index,]
testData = dataset[-index,]
y_test <- matrix(table(1:60,testData$class), nrow=60, ncol=2)

model_knn = train(trainData[2:8], trainData$class, method='knn')
predictions <- predict(object=model_knn, testData[,2:8])
confmtx <- confusionMatrix(predictions, testData$class)
y_pred <- matrix(table(1:60,predictions), nrow=60, ncol=2)
print(model_knn)
print(confmtx)
print(mean(y_test != y_pred))


model_knn_pp = train(trainData[2:8], trainData$class, method='knn', preProcess=c("center","scale"))
predictions_pp <- predict(object=model_knn_pp, testData[,2:8])
confmtx_pp <- confusionMatrix(predictions_pp, testData$class)
y_pred_pp <- matrix(table(1:60,predictions_pp), nrow=60, ncol=2)
print(model_knn_pp)
print(confmtx_pp)
print(mean(y_test != y_pred_pp))


train_control <- trainControl(method="cv", number=10)
grid <- expand.grid(k = c(5,7,9,15,19,21))
model_knn_cv <- train(trainData[2:8], trainData$class, trControl=train_control, method="knn", tuneGrid=grid)
predictions_cv <- predict(object=model_knn_cv, testData[,2:8])
confmtx_cv <- confusionMatrix(predictions_cv, testData$class)
y_pred_cv <- matrix(table(1:60,predictions_cv), nrow=60, ncol=2)
print(model_knn_cv)
print(confmtx_cv)
print(mean(y_test != y_pred_cv))


train_control <- trainControl(method="cv", number=10)
grid <- expand.grid(k = c(5,7,9,15,19,21))
model_knn_cvpp <- train(trainData[2:8], trainData$class, trControl=train_control, method="knn", tuneGrid=grid, preProcess=c("center","scale"))
predictions_cvpp <- predict(object=model_knn_cvpp, testData[,2:8])
confmtx_cvpp <- confusionMatrix(predictions_cvpp, testData$class)
y_pred_cvpp <- matrix(table(1:60,predictions_cvpp), nrow=60, ncol=2)
print(model_knn_cvpp)
print(confmtx_cvpp)
print(mean(y_test != y_pred_cvpp))

