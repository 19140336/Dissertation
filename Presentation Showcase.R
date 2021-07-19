# install.packages("quanteda")
# install.packages("caret")
# install.packages("caretEnsemble")
# install.packages("randomForest")
# install.packages("lattice")
# install.packages("ggplot2")
# #install.packages("fifer")
# install.packages("dplyr")
# install.packages("RWeka")

library(quanteda)
library(caret)
library(caretEnsemble)
library(randomForest)
library(lattice)
library(ggplot2)
#library(fifer)
library(dplyr)
library(RWeka)

library(splitstackshape)
library(tidyverse)
library(SSLR)
library(tidymodels)

# ERROR 'fifer' package no longer exists

set.seed(1234)
#Set working directory
setwd("C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/dataset")

#Reading csv file
data1M <- read.csv(file="data1M.csv", stringsAsFactors=FALSE)

#Creating a corpus
corp <- corpus(data1M$content)

#Attaching the class labels to the corpus as docvars
docvars(corp, "class") = paste0(data1M$class)

#Cleaning the text
data1M_dfm <- dfm(corp, tolower = TRUE, stem = TRUE, remove=stopwords("english"), remove_numbers = TRUE, 
                  remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, remove_hyphens = TRUE,
                  remove_twitter = TRUE, remove_url = TRUE)

#Trim features by term frequency, doc freq, feature chararacter count
data1M_dfm <- dfm_trim(data1M_dfm, min_termfreq = 5, min_docfreq = 10000)
data1M_dfm <- dfm_select(data1M_dfm, min_nchar = 3)

#Apply tf*idf to document feature matrix
data1M_dfm <- dfm_tfidf(data1M_dfm, scheme_tf = "count", scheme_df = "inverse")

#Convert document feature matrix to data.frame
data1M_dfm <- cbind(data.frame(data1M_dfm, docvars(data1M_dfm)))

#Removing the document column that has been added by default
data1M_dfm$document <- NULL

#Split the data into 2 parts: labeled_df (labeled) and testingData (unlabeled)
labeledData <- data1M_dfm[1:2757,]
testingData <- data1M_dfm[2758:nrow(data1M_dfm),]


#Removing objects and free up some memory
rm(data1M, corp, data1M_dfm)
gc()

########## Start of training and validating algorithm performance section ##########
#Splitting our labeled dataset 2757 into 2, for training and for algorithm performance dataset
#Creating ID column to uniquely identify each labeled dataset rows
ID <- c(1:2757)

#Create new dataframe with ID column
labeledData <- cbind(ID,labeledData)

#set.seed(1234)
#Create training dataset with stratified sampling (10%, 30%, 50% and 70%)
train10 <- stratified(labeledData, "class", 0.1)
train30 <- stratified(labeledData, "class", 0.3)
train50 <- stratified(labeledData, "class", 0.5)
train70 <- stratified(labeledData, "class", 0.7)

#Create a not-in operator, not available in R built-in functions
'%nin%' = Negate('%in%')

# Create performance dataset (test data) for algorithm accuracy validation
test10 <- filter(labeledData, ID %nin% train10$ID)
test30 <- filter(labeledData, ID %nin% train30$ID)
test50 <- filter(labeledData, ID %nin% train50$ID)
test70 <- filter(labeledData, ID %nin% train70$ID)

#Removing the ID columns after stratified sampling
train10$ID <- NULL
train30$ID <- NULL
train50$ID <- NULL
train70$ID <- NULL
test10$ID <- NULL
test30$ID <- NULL
test50$ID <- NULL
test70$ID <- NULL

#Randomize training and test performance datasets
train10 <- train10[sample(nrow(train10)),]
train30 <- train30[sample(nrow(train30)),]
train50 <- train50[sample(nrow(train50)),]
train70 <- train70[sample(nrow(train70)),]
test10 <- test10[sample(nrow(test10)),]
test30 <- test30[sample(nrow(test30)),]
test50 <- test50[sample(nrow(test50)),]
test70 <- test70[sample(nrow(test70)),]

########## Start of experimental section ##########

#Create training set with both labelled and unlabelled data
tra = rbind(train70, testingData[1:10000,])
tra$class[which(tra$class == "")] = NA
tra$class = as.numeric(as.factor(tra$class))

# Create testing set
tes = test70
tes$class = as.numeric(as.factor(tes$class))

# Designate base classifier
rf = rand_forest(trees = 100, mode = "classification") %>% set_engine("randomForest")

# Train algorithm
m = selfTraining(learner = rf, perc.full = 0.7, thr.conf = 0.5, max.iter = 10) %>% fit(class ~., data = tra[,-1])

# Make predictions
p = predict(m, tes[,-1])

# Performance metrics (ADD MORE)
sum(p == tes$class) / nrow(p)

