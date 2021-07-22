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
library(dplyr)
library(RWeka)

library(splitstackshape)
library(tidyverse)
library(SSLR)
library(tidymodels)
library(kknn)
library(C50)
library(e1071)

set.seed(1234)
#Set working directory
setwd("C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/dataset")

#Read csv file
data1M <- read.csv(file="data1M.csv", stringsAsFactors=FALSE)

#Create corpus
corp <- corpus(data1M$content)

#Attach class labels to the corpus as docvars
docvars(corp, "class") = paste0(data1M$class)

#Clean the text
data1M_dfm <- dfm(corp, tolower = TRUE, stem = TRUE, remove=stopwords("english"), remove_numbers = TRUE, 
                  remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, remove_hyphens = TRUE,
                  remove_twitter = TRUE, remove_url = TRUE)

#Trim features by term frequency, document frequency, feature character count
data1M_dfm <- dfm_trim(data1M_dfm, min_termfreq = 5, min_docfreq = 10000)
data1M_dfm <- dfm_select(data1M_dfm, min_nchar = 3)

#Apply tf-idf to document feature matrix
data1M_dfm <- dfm_tfidf(data1M_dfm, scheme_tf = "count", scheme_df = "inverse")

#Convert document feature matrix to data frame
data1M_dfm <- cbind(data.frame(data1M_dfm, docvars(data1M_dfm)))

#Remove the document column that has been added by default
data1M_dfm$document <- NULL

#Split the data into labelled and unlabelled
labelledData <- data1M_dfm[1:2757,]
unlabelledData <- data1M_dfm[2758:nrow(data1M_dfm),]

#Remove objects and free up some memory
rm(data1M, corp, data1M_dfm)
gc()

########## Create training and testing data section ##########
#Splitting our labelled dataset 2757 into 2, for training and for algorithm performance dataset
#Creating ID column to uniquely identify each labelled dataset rows
ID <- c(1:2757)

#Create new dataframe with ID column
labelledData <- cbind(ID,labelledData)

#Create training dataset with stratified sampling
tra <- stratified(labelledData, "class", 0.7)

#Create a not-in operator, not available in R built-in functions
'%nin%' = Negate('%in%')

# Create testing dataset
tes <- filter(labelledData, ID %nin% tra$ID)

#Remove ID columns after stratified sampling
tra$ID <- NULL
tes$ID <- NULL

#Randomize training and test performance dataset
tra <- tra[sample(nrow(tra)),]
tes <- tes[sample(nrow(tes)),]

# #Create training set for semi-supervised training
# unlabelledData = unlabelledData[sample(nrow(unlabelledData)),]
# semitra = rbind(tra, unlabelledData[1:100000,])
# semitra$class[which(semitra$class == "")] = NA

# Change classes to number
tra$class = as.factor(as.numeric(as.factor(tra$class)))
tes$class = as.factor(as.numeric(as.factor(tes$class)))

#Create training set with both labelled and unlabelled data
semitra = rbind(tra, unlabelledData[1:1000,])
semitra$class[which(semitra$class == "")] = NA
semitra$class = as.numeric(as.factor(semitra$class))

#Clear up memory
rm(labelledData, unlabelledData)
gc()

########## Experiment section ##########

#Designate base classifier
dectree = decision_tree(mode = "classification") %>% set_engine("C5.0")

#Train base classifier
dectreefit = fit(object = dectree, class~., data = tra[,-1])

#Make predictions with base classifier
basepred = predict(dectreefit, tes[,-1])

# Train wrapper algorithm
selffit = selfTraining(learner = dectree, perc.full = 0.7, thr.conf = 0.5, max.iter = 10) %>% fit(class ~., data = semitra[,-1])

# Make predictions with wrapper algorithm
selfpred = predict(selffit, tes[,-1])

# Performance metrics
confusionMatrix(as.factor(as.numeric(unlist(tes$class))), as.factor(as.numeric(unlist(basepred))))
confusionMatrix(as.factor(as.numeric(unlist(tes$class))), as.factor(as.numeric(unlist(selfpred))))
