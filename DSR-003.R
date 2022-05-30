# ------------------------------
# SETScholars: DSR-003.R ------
# ------------------------------
############################################################################
# This is an end-2-end Applied Machine Learning Script using R and MySQL
# Title: Applied Machine Learning in R: IRIS Flower Classification 
# Knowledge required: Basic R, CARET package and MySQL
# System requirements:
#   a) R (3.X) distribution
#   b) MySQL 5.7 with an user: root and password:
############################################################################

#@author: 
#Nilimesh Halder, PhD
#BSc in Computer Science and Engineering, 
#@ Khulna University, Bangladesh.
#PhD in Artificial Intelligence and Applied Machine Learning, 
#@ The University of Western Australia, Australia.

# -----------------------------------------------------------------------------
# Steps in Applied Machine Learning:
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

###############################################
## Load necessary Libraries and Dataset #######
###############################################
library(DBI)
library(RMySQL)
library(corrgram)
library(caret)
library(doMC)

# register cores for parallel processing
registerDoMC(cores=4)

# get current ditecroty
getwd()
# set working directory where CSV data file is located
setwd("/Users/nilimesh/Desktop/Data Science Recipes/DSR-3-R")

# Load the DataSets
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

# Check Data types of individual column
data.class(dataSet$sepal_length)
data.class(dataSet$sepal_width)
data.class(dataSet$petal_length)
data.class(dataSet$petal_width)
data.class(dataSet$class)

#######################################
## Connect to a MySQL Database ########
#######################################

# create a MySQL driver 
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306
myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword, dbname= myDbname, port= myPort)
if(dbIsValid(con)) {
  print('MySQL Connection is Successful')
} else {print('MySQL Connection is Unsuccessful')}

# Export DataFrame to a MySQL table 
response <- dbWriteTable(conn = con, name = 'irisdata', value = dataSet, 
                         row.names = FALSE, overwrite = TRUE)

if(response) {print('Data import is successful')
} else {print('Data import is unsuccessful')}

## Write a query here and execute it to retrive data from MySQL Database
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

###############################################################
## Check dataset that retrived from MySQL database ############
###############################################################

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

###############################################################################
## Exploring or Summarising dataset with descriptive statistics ###############
###############################################################################

# Find out if there is missing value
rowSums(is.na(dataset))
colSums(is.na(dataset))

# Missing data treatment if exists
#dataset[dataset$columnName=="& ","columnName"] <- NA 
#drop columns
#dataset <- within(dataset, rm(columnName))

##########################################
# Summary of dataset #####################
##########################################

# USING lapply - When you want to apply a function to each element of a list in turn and get a list back.
lapply(dataset[1:12], FUN = sum)
lapply(dataset[1:12], FUN = mean)
lapply(dataset[1:12], FUN = median)
lapply(dataset[1:12], FUN = min)
lapply(dataset[1:12], FUN = max)
lapply(dataset[1:12], FUN = length)

# USING sapply - When you want to apply a function to each element of a list in turn, 
# but you want a vector back, rather than a list.
sapply(dataset[1:12], FUN = sum)
sapply(dataset[1:12], FUN = mean)
sapply(dataset[1:12], FUN = median)
sapply(dataset[1:12], FUN = min)
sapply(dataset[1:12], FUN = max)
sapply(dataset[1:12], FUN = length)

# USING tapply - For when you want to apply a function to subsets of a vector 
# and the subsets are defined by some other vector, usually a factor.
tapply(dataset$sepal_length, dataset$class, FUN = summary)
tapply(dataset$sepal_width, dataset$class, FUN = summary)
tapply(dataset$petal_length, dataset$class, FUN = summary)
tapply(dataset$petal_width, dataset$class, FUN = summary)

# USING Aggregate FUNCTION
aggregate(dataset$sepal_length, list(dataset$class), summary)
aggregate(dataset$sepal_width, list(dataset$class), summary)
aggregate(dataset$petal_length, list(dataset$class), summary)
aggregate(dataset$petal_width, list(dataset$class), summary)

# USING "by"
by(dataset[1:12], dataset[13], FUN = summary)
by(dataset[1:12], dataset$class, FUN = summary)

######################################
## Visualisation of DataSet ##########
######################################

# Print Column Names
colnames(dataset)

# Print Data Types of each column
for(i in 1:length(dataset)) {
  print(data.class(dataset[,i]))
}

# Histogram
par(mfrow=c(2,2))
x <- dataset$sepal_length
hist(x,  xlab = "Sapel Length", ylab = "Count", main = "")
x <- dataset$sepal_width
hist(x,  xlab = "Sapel Width", ylab = "Count", main = "")
x <- dataset$petal_length
hist(x,  xlab = "petal Length", ylab = "Count", main = "")
x <- dataset$petal_width
hist(x,  xlab = "Petal Width", ylab = "Count", main = "")

# Histogram with Density graph
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

# Barplot of categorical data
par(mfrow=c(2,2))
barplot(table(dataset$class), ylab = "Count", 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$class)), ylab = "Proportion", 
        col=c("darkblue","red", "green"))
barplot(table(dataset$class), xlab = "Count", horiz = TRUE, 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$class)), xlab = "Proportion", horiz = TRUE, 
        col=c("darkblue","red", "green"))

# Box Plot of Numerical Data
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

# Scatter Plots
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

# Corelation Diagram using "corrgram" package
x <- dataset[1:4]
#x is a data frame with one observation per row.
corrgram(x)
#order=TRUE will cause the variables to be ordered using principal component analysis of the correlation matrix.
corrgram(x, order = TRUE)
# lower.panel= and upper.panel= to choose different options below and above the main diagonal respectively. 
# (the filled portion of the pie indicates the magnitude of the correlation)
# lower.panel=  
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.shade, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.ellipse, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.pts, upper.panel = NULL)
#off diagonal panels
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

# Pie Chart
par(mfrow=c(1,1))
x <- table(dataset$class)
lbls <- paste(names(x), "\nTotal Count:", x, sep="")
pie(x, labels = lbls, main="Pie Chart of Different Classes", col = c("red","blue","green"))

##################################################################
## Pre-Processing of DataSet i.e. train : test split #############
##################################################################

train_test_index <- createDataPartition(dataset$class, p=0.67, list=FALSE)
training_dataset <- dataset[train_test_index,]
testing_dataset <- dataset[-train_test_index,]

#################################################################################
## Evaluating Algorithm i.e. training, testing and evaluation ###################
#################################################################################

# Check available ML Algorithms
names(getModelInfo())

# cross Validation setup
control <- trainControl(method="cv", number=2, verbose = TRUE)

# Machine Learning Algorithm and parameter settings
# Training using lda : Linear Discriminant Analysis
fit.lda <- train(class~., data=training_dataset, method="lda", metric="Accuracy", trControl=control)

# Testing model skill on validation dataset
predictions <- predict(fit.lda, newdata=testing_dataset)

# Evaluation of Trained Model
confusionMatrix(predictions, testing_dataset$class)

######################################
## Save the model to disk ############
######################################

final_model <- fit.lda
saveRDS(final_model, "./final_model.rds")

##########################################################
## Finalise the trained model and make prediction ########
##########################################################

# Connecting a MySQL Database
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306
myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword, dbname= myDbname, port= myPort)

if(dbIsValid(con)) {
  print('MySQL Connection is Successful')
} else {print('MySQL Connection is Unsuccessful')}

## Write a query here and execute it to retrive data from MySQL Database
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

## Load the trained model from disk
trained_model <- readRDS("./final_model.rds")
print(trained_model)

# make a predictions on "new data" using the final model
final_predictions <- predict(trained_model, dataset)

# Save result in a CSV file and/ or MySQL Table
result <- data.frame(final_predictions)
dim(result)
dim(dataset)

# merge prediction with dataset 
finalResult <- cbind(dataset, result)
dim(finalResult)

# in CSV file
cols <- c(1:4, 13)
write.csv(finalResult[cols], file = "finalResult.csv", row.names = FALSE)

# in MySQL Table
dbWriteTable(conn = con, name = 'irisresult', value = finalResult[cols], 
             row.names = FALSE, overwrite = TRUE)
dbDisconnect(conn = con)

print("End 2 End Applied Machine Learning and Data Science Recipes")