# ------------------------------
# SETScholars: DSR-010.R ------
# ------------------------------

############################################################################
# This is an end-2-end Applied Machine Learning Script using R and MySQL

# Title: Supervised Learning (Classification) using KNN and LVQ in R: 
# Applied Machine Learning Recipe-10

# Knowledge required: Basic R, CARET package and MySQL
# System requirements:
#   a) R (3.X) distribution
#   b) MySQL 5.7 with an user: root and password: root888
############################################################################

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Author: 
# Nilimesh Halder, PhD
# BSc in Computer Science and Engineering, 
#           @Khulna University, Bangladesh.
# PhD in Artificial Intelligence and Applied Machine Learning, 
#           @The University of Western Australia, Australia.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Steps in Applied Machine Learning & Predictive Modelling:
# 1. Load Library
# 2. Load Dataset to which Machine Learning Algorithm to be applied
#    Either a) load from a CSV file or b) load from a Database   
# 3. Summarisation of Data to understand dataset (Descriptive Statistics)
# 4. Visualisation of Data to understand dataset (Plots, Graphs etc.)
# 5. Data pre-processing & Data transformation (split into train-test datasets)
# 6. Application of a Machine Learning Algorithm to training dataset 
#   a) setup a ML algorithm and parameter settings
#   b) cross validation setup with training dataset
#   c) training & fitting Algorithm with training Dataset
#   d) evaluation of trained Algorithm (or Model) and result
#   e) saving the trained model for future prediction
# 7. Finalise the trained model and make prediction            
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


##########################################################################
# 1. Load Library
##########################################################################
# Load necessary Libraries
library(DBI)
library(RMySQL)
library(corrgram)
library(caret)
library(doMC)

# --------------------------------------
# register cores for parallel processing
# --------------------------------------
registerDoMC(cores=4)

# --------------------------------------
# get current ditecroty
# --------------------------------------
getwd()

# -----------------------------------------------------
# set working directory where CSV data file is located
# -----------------------------------------------------
setwd("/Users/nilimesh/Desktop/Data Science Recipes/DSR-4-R")


##########################################################################
# 2. Load Dataset to which Machine Learning Algorithm to be applied
#    Either a) load from a CSV file or b) load from a Database   
##########################################################################
# --------------------------------------
# Load the DataSets from a CSV file & 
# Set column names 
# --------------------------------------
dataSet <- read.csv("iris.data.csv", header = FALSE, sep = ',')
colnames(dataSet) <- c('sepal_length', 'sepal_width', 'petal_length', 
                       'petal_width', 'class')

# Print top 10 rows in the dataSet
head(dataSet, 10)

# Print last 10 rows in the dataSet
tail(dataSet, 10)

# Dimention of Dataset
dim(dataSet)

# Check Data types of each column
table(unlist(lapply(dataSet, class)))

#Check column names
colnames(dataSet)

# --------------------------------------
# Check Data types of individual column
# --------------------------------------
data.class(dataSet$sepal_length)
data.class(dataSet$sepal_width)
data.class(dataSet$petal_length)
data.class(dataSet$petal_width)
data.class(dataSet$class)

# --------------------------------------
# Connect to a MySQL Database
# --------------------------------------
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306
myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword, 
                dbname= myDbname, port= myPort)

# check whether MySQL connection is successful or not
if(dbIsValid(con)) {
              print('MySQL Connection is Successful')
} else {
              print('MySQL Connection is Unsuccessful')
}

# --------------------------------------
# Export DataFrame to a MySQL table 
# --------------------------------------
response <- dbWriteTable(conn = con, name = 'irisdata', value = dataSet, 
                         row.names = FALSE, overwrite = TRUE)
if(response) {print('Data export is successful')
} else {print('Data export is unsuccessful')}

# ------------------------------------------------------------------------
# Write a query here and execute it to retrive data from MySQL Database
# ------------------------------------------------------------------------
sql = 'SELECT sepal_length, 
              sepal_width, 
              petal_length, 
              petal_width, 
              round(sepal_length/sepal_width,2) as ratio1, 
              round(sepal_width/petal_length,2) as ratio2,
              round(petal_length/petal_width,2) as ratio3,
              round(petal_width/sepal_length,2) as ratio4,
              round(sepal_width/sepal_length,2) as ratio5, 
              round(petal_length/sepal_width,2) as ratio6,
              round(petal_width/petal_length,2) as ratio7,
              round(sepal_length/petal_width,2) as ratio8,
              class 
        FROM irisdata;'

result = dbSendQuery(conn = con, statement = sql)
dataset <- dbFetch(res = result)
dbClearResult(result)
dbDisconnect(conn = con)

# --------------------------------------------------
# Check dataset that retrived from MySQL database
# --------------------------------------------------
# Print top 10 rows in the dataSet
head(dataset, 10)
# Print last 10 rows in the dataSet
tail(dataset, 10)
# Dimention of Dataset
dim(dataset)
# Check Data types of each column
table(unlist(lapply(dataset, class)))
#Check column names
colnames(dataset)

##########################################################################
# 3. Summarisation of Data to understand dataset (Descriptive Statistics)
##########################################################################
# --------------------------------------------------------------
# Exploring or Summarising dataset with Descriptive Statistics
# --------------------------------------------------------------
# Find out if there is missing value
rowSums(is.na(dataset))
colSums(is.na(dataset))

# Missing data treatment if exists
#dataset[dataset$columnName=="& ","columnName"] <- NA 
#drop columns
#dataset <- within(dataset, rm(columnName))

# --------------------------------------
# Summary of dataset
# --------------------------------------

# -----------------------------------------------------
# lapply - When you want to apply a function to 
# each element of a list in turn and get a list back.
# -----------------------------------------------------
lapply(dataset[1:12], FUN = sum)
lapply(dataset[1:12], FUN = mean)
lapply(dataset[1:12], FUN = median)
lapply(dataset[1:12], FUN = min)
lapply(dataset[1:12], FUN = max)
lapply(dataset[1:12], FUN = length)

# -------------------------------------------------
# sapply - When you want to apply a function to 
# each element of a list in turn, 
# but you want a vector back, rather than a list.
# -------------------------------------------------
sapply(dataset[1:12], FUN = sum)
sapply(dataset[1:12], FUN = mean)
sapply(dataset[1:12], FUN = median)
sapply(dataset[1:12], FUN = min)
sapply(dataset[1:12], FUN = max)
sapply(dataset[1:12], FUN = length)

# ----------------------------------------------------------------------
# tapply - For when you want to apply a function to subsets of a vector 
# and the subsets are defined by some other vector, usually a factor.
# ----------------------------------------------------------------------
tapply(dataset$sepal_length, dataset$class, FUN = summary)
tapply(dataset$sepal_width, dataset$class, FUN = summary)
tapply(dataset$petal_length, dataset$class, FUN = summary)
tapply(dataset$petal_width, dataset$class, FUN = summary)

# --------------------------------------
# Using Aggregate FUNCTION
# --------------------------------------
aggregate(dataset$sepal_length, list(dataset$class), summary)
aggregate(dataset$sepal_width, list(dataset$class), summary)
aggregate(dataset$petal_length, list(dataset$class), summary)
aggregate(dataset$petal_width, list(dataset$class), summary)

# --------------------------------------
# Using "by"
# --------------------------------------
by(dataset[1:12], dataset[13], FUN = summary)
by(dataset[1:12], dataset$class, FUN = summary)

#########################################################################
# 4. Visualisation of Data to understand dataset (Plots, Graphs etc.)
#########################################################################
# --------------------------------------
# Visualising DataSet
# --------------------------------------
# Print Column Names
colnames(dataset)

# Print Data Types of each column
for(i in 1:length(dataset)) {
  print(data.class(dataset[,i]))
}

# --------------------------------------
# Histogram
# --------------------------------------
par(mfrow=c(2,2))
x <- dataset$sepal_length
hist(x,  xlab = "Sapel Length", ylab = "Count", main = "")
x <- dataset$sepal_width
hist(x,  xlab = "Sapel Width", ylab = "Count", main = "")
x <- dataset$petal_length
hist(x,  xlab = "petal Length", ylab = "Count", main = "")
x <- dataset$petal_width
hist(x,  xlab = "Petal Width", ylab = "Count", main = "")

# --------------------------------------
# Histogram with Density graph
# --------------------------------------
par(mfrow=c(2,2))

x <- dataset$sepal_length
h <- hist(x,  xlab = "Sapel Length", ylab = "Count", ylim=c(0,40), main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$sepal_width
h <- hist(x,  xlab = "Sapel Width", ylab = "Count", ylim=c(0,40), main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$petal_length
h <- hist(x,  xlab = "Petal Length", ylab = "Count", ylim=c(0,40), main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$petal_width
h <- hist(x,  xlab = "Petal Width", ylab = "Count", ylim=c(0,40), main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

# --------------------------------------
# Barplot of categorical data
# --------------------------------------
par(mfrow=c(2,2))
barplot(table(dataset$class), ylab = "Count", 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$class)), ylab = "Proportion", 
        col=c("darkblue","red", "green"))
barplot(table(dataset$class), xlab = "Count", horiz = TRUE, 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$class)), xlab = "Proportion", horiz = TRUE, 
        col=c("darkblue","red", "green"))

# --------------------------------------
# Box Plot of Numerical Data
# --------------------------------------
par(mfrow=c(1,4))
boxplot(dataset$sepal_length, ylab = "Sapel Length")
boxplot(dataset$sepal_width, ylab = "Sapel Width")
boxplot(dataset$petal_length, ylab = "Petal Length")
boxplot(dataset$petal_width, ylab = "Petal Width")

par(mfrow=c(1,4))
boxplot(dataset$ratio1, ylab = "Ratio 1")
boxplot(dataset$ratio2, ylab = "Ratio 2")
boxplot(dataset$ratio3, ylab = "Ratio 3")
boxplot(dataset$ratio4, ylab = "Ratio 4")

par(mfrow=c(1,4))
boxplot(dataset$ratio5, ylab = "Ratio 5")
boxplot(dataset$ratio6, ylab = "Ratio 6")
boxplot(dataset$ratio7, ylab = "Ratio 7")
boxplot(dataset$ratio8, ylab = "Ratio 8")

# --------------------------------------
# Scatter Plots
# --------------------------------------
par(mfrow=c(2,2))
plot(dataset$sepal_length, pch = 20)
plot(dataset$petal_length, pch = 20)
plot(dataset$sepal_width, pch = 20)
plot(dataset$petal_width, pch = 20)

par(mfrow=c(2,2))
plot(dataset$sepal_length, dataset$sepal_width, pch = 20)
plot(dataset$petal_length, dataset$petal_width, pch = 20)
plot(dataset$petal_length, dataset$sepal_length, pch = 20)
plot(dataset$sepal_width,  dataset$petal_width, pch = 20)

# -------------------------------------------------------
# Corelation Diagram using "corrgram" package
# -------------------------------------------------------
x <- dataset[1:4]

# x is a data frame with one observation per row.
corrgram(x)

# order=TRUE will cause the variables to be ordered using principal component analysis of the correlation matrix.
corrgram(x, order = TRUE)

# lower.panel= and upper.panel= to choose different options below and above the main diagonal respectively. 

# (the filled portion of the pie indicates the magnitude of the correlation)
# lower.panel=  
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.shade, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.ellipse, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.pts, upper.panel = NULL)

# off diagonal panels
# lower.panel= & upper.panel=
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts)
corrgram(x, order = TRUE, lower.panel = panel.shade, upper.panel = panel.pie)
corrgram(x, order = TRUE, lower.panel = panel.ellipse, upper.panel = panel.shade)
corrgram(x, order = TRUE, lower.panel = panel.pts, upper.panel = panel.pie)

# upper.panel=
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.pts)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.pie)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.shade)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.ellipse)

#text.panel= and diag.panel= refer to the main diagnonal. 
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts,
         text.panel=panel.txt)
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts,
         text.panel=panel.txt, diag.panel=panel.minmax,
         main="Correlation Diagram")

# --------------------------------------
# Pie Chart
# --------------------------------------
par(mfrow=c(1,1))
x <- table(dataset$class)
lbls <- paste(names(x), "\nTotal Count:", x, sep="")
pie(x, labels = lbls, main="Pie Chart of Different Classes", col = c("red","blue","green"))

################################################################################
# 5. Data pre-processing & Data transformation (split into train-test datasets)
################################################################################
# -----------------------------------------------------
# Pre-Processing of DataSet i.e. train : test split
# -----------------------------------------------------
train_test_index <- createDataPartition(dataset$class, p=0.67, list=FALSE)
training_dataset <- dataset[train_test_index,]
testing_dataset <- dataset[-train_test_index,]

################################################################################
# 6. Application of a Machine Learning Algorithm to training dataset 
#   a) setup a ML algorithm and parameter settings
#   b) cross validation setup with training dataset
#   c) training & fitting Algorithm with training Dataset
#   d) evaluation of trained Algorithm (or Model) and result
#   e) saving the trained model for future prediction
################################################################################

# -------------------------------------------------------------
# Evaluating Algorithm i.e. training, testing and evaluation
# -------------------------------------------------------------
# Check available ML Algorithms
names(getModelInfo())

# Get information of a particular ML Algorithm / Method (e.g. svmLinear)
getModelInfo("svmLinear")

# -------------------------------------------------------------
# cross Validation setup
# -------------------------------------------------------------
control <- trainControl(method="cv", number=10, verbose = TRUE)

# -------------------------------------------------------------
##############################################################
# Machine Learning Algorithm and parameter settings
##############################################################
# -------------------------------------------------------------
# LDA # Training using lda : Linear Discriminant Analysis
# -------------------------------------------------------------
fit.lda <- train(class~., data=training_dataset, method="lda",
                 metric="Accuracy", trControl=control)
print(fit.lda)
# Testing model skill on validation dataset
predictions_LDA <- predict(fit.lda, newdata=testing_dataset)
# -------------------------------------------------------------
# NB # Training using NB : Naive Bayes
# -------------------------------------------------------------
fit.nb <- train(class~., data=training_dataset, method="nb",
                metric="Accuracy", trControl=control)
print(fit.nb)
# Testing skill on validation dataset
predictions_NB <- predict(fit.nb, newdata=testing_dataset)
# -------------------------------------------------------------
# KNN # Training using KNN : K nearest neighbourhood
# -------------------------------------------------------------
fit.knn <- train(class~., data=training_dataset, method="knn",
                 metric="Accuracy", trControl=control)
print(fit.knn)
# Testing skill on validation dataset
predictions_KNN <- predict(fit.knn, newdata=testing_dataset)
# -------------------------------------------------------------
# LogitBoost # Training using LogitBoost : Classification
# -------------------------------------------------------------
fit.logitBoost <- train(class~., data=training_dataset, method="LogitBoost",
                        metric="Accuracy", trControl=control)
print(fit.logitBoost)
# Testing skill on validation dataset
predictions_LB <- predict(fit.logitBoost, newdata=testing_dataset)
# -------------------------------------------------------------
# SVM # Training using svmLinear : Classification
# -------------------------------------------------------------
fit.svmLinear <- train(class~., data=training_dataset, method="svmLinear",
                       metric="Accuracy", trControl=control)
print(fit.svmLinear)
# Testing skill on validation dataset
predictions_SVM <- predict(fit.svmLinear, newdata=testing_dataset)
# -------------------------------------------------------------
# Collect resampling statistics across ALL trained models
# -------------------------------------------------------------
results <- resamples(list(LDA = fit.lda,
                          NB  = fit.nb,
                          KNN = fit.knn,
                          LB  = fit.logitBoost,
                          SVM = fit.svmLinear))
# -------------------------------------------------------------
# summarize results
# -------------------------------------------------------------
summary(results)
# -------------------------------------------------------------
# plot the results
# -------------------------------------------------------------
dotplot(results)
# -------------------------------------------------------------
# Evaluation of Trained Model
# -------------------------------------------------------------
res_LDA  <- confusionMatrix(predictions_LDA, testing_dataset$class)
res_LB   <- confusionMatrix(predictions_LB, testing_dataset$class)
res_NB   <- confusionMatrix(predictions_NB, testing_dataset$class)
res_KNN  <- confusionMatrix(predictions_KNN, testing_dataset$class)
res_SVM  <- confusionMatrix(predictions_SVM, testing_dataset$class)
print("Results from LDA ... ..."); print(res_LDA); print(res_LDA$overall)
print("Results from LB ... ...");  print(res_LB);  print(res_LB$overall)
print("Results from NB ... ...");  print(res_NB);  print(res_NB$overall)
print("Results from KNN ... ..."); print(res_KNN); print(res_KNN$overall)
print("Results from SVM ... ..."); print(res_SVM); print(res_SVM$overall)
# -------------------------------------------------------------
# Save the model to disk
# -------------------------------------------------------------
final_model <- fit.lda;        saveRDS(final_model, "./final_model_LDA.rds")
final_model <- fit.logitBoost; saveRDS(final_model, "./final_model_LB.rds")
final_model <- fit.svmLinear;  saveRDS(final_model, "./final_model_SVM.rds")
final_model <- fit.knn;        saveRDS(final_model, "./final_model_KNN.rds")
final_model <- fit.nb;         saveRDS(final_model, "./final_model_NB.rds")
#######################################################
# 7. Finalise the trained model and make prediction
#######################################################
# -------------------------------------------------------------
# Connecting a MySQL Database
# -------------------------------------------------------------
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306

myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword,
                dbname= myDbname, port= myPort)
if(dbIsValid(con)) {
  print('MySQL Connection is Successful')
} else {print('MySQL Connection is Unsuccessful')}
# ------------------------------------------------------------------------
# Write a query here and execute it to retrive data from MySQL Database
# ------------------------------------------------------------------------
sql = 'SELECT sepal_length,
sepal_width,
petal_length,
petal_width,
round(sepal_length/sepal_width,2) as ratio1,
round(sepal_width/petal_length,2) as ratio2,
round(petal_length/petal_width,2) as ratio3,
round(petal_width/sepal_length,2) as ratio4,
round(sepal_width/sepal_length,2) as ratio5,
round(petal_length/sepal_width,2) as ratio6,
round(petal_width/petal_length,2) as ratio7,
round(sepal_length/petal_width,2) as ratio8
FROM irisdata;'
result = dbSendQuery(conn = con, statement = sql)
dataset <- dbFetch(res = result)
dbClearResult(result)
dim(dataset)
# -------------------------------------------------------------
# Load the trained model from disk
# -------------------------------------------------------------
trained_model <- readRDS("./final_model_KNN.rds")
print(trained_model)
# -------------------------------------------------------------
# make a predictions on "new data" using the final model
# -------------------------------------------------------------
final_predictions <- predict(trained_model, dataset)
# -------------------------------------------------------------
# Save result in a CSV file and/ or MySQL Table
# -------------------------------------------------------------
result <- data.frame(final_predictions)
dim(result)
dim(dataset)
# -------------------------------------------------------------
# merge prediction with dataset
# -------------------------------------------------------------
finalResult <- cbind(dataset, result)
dim(finalResult)
# -------------------------------------------------------------
# Write Results in CSV file
# -------------------------------------------------------------
cols <- c(1:4, 13)
write.csv(finalResult[cols], file = "finalResult.csv", row.names = FALSE)
# -------------------------------------------------------------
# Write Results in MySQL Table

# -------------------------------------------------------------
dbWriteTable(conn = con, name = 'irisresult', value = finalResult[cols],
             row.names = FALSE, overwrite = TRUE)
dbDisconnect(conn = con)
# KAPPA Interpretation :
# In plain English,
# it measures how much better the classier is comparing with guessing
# with the target distribution.
# Poor agreement = 0.20 or less
# Fair agreement = 0.20 to 0.40
# Moderate agreement = 0.40 to 0.60
# Good agreement = 0.60 to 0.80
# Very good agreement = 0.80 to 1.00

# -------------------------------------------------------------
# Disclaimer --------------------------------------------------
# -------------------------------------------------------------
# The information and codes presented within this recipe 
# is only for educational and coaching purposes for beginners
# and app-developers. Anyone can practice and apply 
# the recipe presented here, but the reader is taking 
# full responsibility for his/her actions.
# The author of this recipe (code / program) has made every effort 
# to ensure the accuracy of the information was correct at time 
# of publication. 
# The author does not assume and hereby disclaims any liability 
# to any party for any loss, damage, or disruption caused by errors 
# or omissions, whether such errors or omissions result from accident, 
# negligence, or any other cause. Some of the information presented 
# here could be also found in public knowledge domains. 
# -------------------------------------------------------------


# -----------------------------------------------------------------
# An End-2-End Applied Machine Learning & Data Science Recipe in R 
# -----------------------------------------------------------------
