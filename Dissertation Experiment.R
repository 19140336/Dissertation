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
library(pROC)
library(tictoc)

#Record the duration of the experiment
tic.clearlog()
tic("Timer", msg = "Total time")

set.seed(1234)

#Read csv file
data1M = read.csv(file="C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/dataset/data1M.csv", stringsAsFactors=FALSE)

#Create corpus
corp = corpus(data1M$content)

#Attach class labels to the corpus as docvars
docvars(corp, "class") = paste0(data1M$class)

#Clean the text
data1M_dfm = dfm(corp, tolower = TRUE, stem = TRUE, remove=stopwords("english"), remove_numbers = TRUE, 
                  remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, remove_hyphens = TRUE,
                  remove_twitter = TRUE, remove_url = TRUE)

#Trim features by term frequency, document frequency, feature character count
data1M_dfm = dfm_trim(data1M_dfm, min_termfreq = 5, min_docfreq = 10000)
data1M_dfm = dfm_select(data1M_dfm, min_nchar = 3)

#Apply tf-idf to document feature matrix
data1M_dfm = dfm_tfidf(data1M_dfm, scheme_tf = "count", scheme_df = "inverse")

#Convert document feature matrix to data frame
data1M_dfm = cbind(data.frame(data1M_dfm, docvars(data1M_dfm)))

#Remove the document column that has been added by default
data1M_dfm$document = NULL

#Split the data into labelled and unlabelled
labelledData = data1M_dfm[1:2757,]
unlabelledData = data1M_dfm[2758:nrow(data1M_dfm),]

#Remove objects and free up some memory
rm(data1M, corp, data1M_dfm)
gc()

########## Create training and testing data section ##########
#Splitting our labelled dataset 2757 into 2, for training and for algorithm performance dataset

#Create folds with stratified sampling
folds = createFolds(factor(labelledData$class), k = 10, list = T)
semifolds = c()
for (i in 1:10){
  semifolds = c(semifolds, list(sample(nrow(unlabelledData), 2000)))
}

#Designate base classifiers
dectree = decision_tree(mode = "classification") %>% set_engine("C5.0")
svmrbf = svm_rbf(mode = "classification") %>% set_engine("kernlab")
knn = nearest_neighbor(mode = "classification") %>% set_engine("kknn")
logreg = logistic_reg(mode = "classification") %>% set_engine("glm")

#Define confusion matrix list
basep1 = c()
basep2 = c()
basep3 = c()
basep4 = c()
selfp1 = c()
selfp2 = c()
selfp3 = c()
selfp4 = c()

# Change classes to number
labelledData$class = as.factor(as.numeric(as.factor(labelledData$class)))
unlabelledData$class = as.factor(rep(NA, nrow(unlabelledData)))

#Initialize confusion matrix list
basep1 = vector("list", 10)
basep2 = vector("list", 10)
basep3 = vector("list", 10)
basep4 = vector("list", 10)

selfp1 = vector("list", 10)
selfp2 = vector("list", 10)
selfp3 = vector("list", 10)
selfp4 = vector("list", 10)

#Perform cross validation
for (i in 1:10){
  #Define training and testing sets
  tes = labelledData[folds[[i]],]
  tra = labelledData[-folds[[i]],]
  semitra = rbind(labelledData[-folds[[i]],], unlabelledData[semifolds[[i]],])
  
  #Train using base classifiers
  basefit1 = fit(object = dectree, class~., data = tra[,-1])
  basefit2 = fit(object = svmrbf, class~., data = tra[,-1])
  basefit3 = fit(object = knn, class~., data = tra[,-1])
  basefit4 = fit(object = logreg, class~., data = tra[,-1])
  
  #Predict using base classifiers
  basepred1 = predict(basefit1, tes[,-1])
  basepred2 = predict(basefit2, tes[,-1])
  basepred3 = predict(basefit3, tes[,-1])
  basepred4 = predict(basefit4, tes[,-1])
  
  #Train wrapper algorithms
  selffit1 = selfTraining(learner = dectree, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = semitra[,-1])
  selffit2 = selfTraining(learner = svmrbf, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = semitra[,-1])
  selffit3 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = semitra[,-1])
  selffit4 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = semitra[,-1])
  
  # Make predictions with wrapper algorithm
  selfpred1 = predict(selffit1, tes[,-1])
  selfpred2 = predict(selffit2, tes[,-1])
  selfpred3 = predict(selffit3, tes[,-1])
  selfpred4 = predict(selffit4, tes[,-1])
  
  #Record confusion matrices and auc values (Note: am going to finish this)
  if (i == 1){
    basep1 = confusionMatrix(as.factor(as.numeric(unlist(basepred1))), tes$class)$table
    basep2 = confusionMatrix(as.factor(as.numeric(unlist(basepred2))), tes$class)$table
    basep3 = confusionMatrix(as.factor(as.numeric(unlist(basepred3))), tes$class)$table
    basep4 = confusionMatrix(as.factor(as.numeric(unlist(basepred4))), tes$class)$table
    
    selfp1 = confusionMatrix(as.factor(as.numeric(unlist(selfpred1))), tes$class)$table
    selfp2 = confusionMatrix(as.factor(as.numeric(unlist(selfpred2))), tes$class)$table
    selfp3 = confusionMatrix(as.factor(as.numeric(unlist(selfpred3))), tes$class)$table
    selfp4 = confusionMatrix(as.factor(as.numeric(unlist(selfpred4))), tes$class)$table
    
    baseauc1 = multiclass.roc(tes$class, as.numeric(unlist(basepred1)))$auc[1]
    baseauc2 = multiclass.roc(tes$class, as.numeric(unlist(basepred1)))$auc[1]
    baseauc3 = multiclass.roc(tes$class, as.numeric(unlist(basepred1)))$auc[1]
    baseauc4 = multiclass.roc(tes$class, as.numeric(unlist(basepred1)))$auc[1]
    
    selfauc1 = multiclass.roc(tes$class,as.numeric(unlist(selfpred1)))$auc[1]
    selfauc2 = multiclass.roc(tes$class,as.numeric(unlist(selfpred2)))$auc[1]
    selfauc3 = multiclass.roc(tes$class,as.numeric(unlist(selfpred3)))$auc[1]
    selfauc4 = multiclass.roc(tes$class,as.numeric(unlist(selfpred4)))$auc[1]
    
  } else{
    basep1 = (basep1 + confusionMatrix(as.factor(as.numeric(unlist(basepred1))), tes$class)$table) / 2
    basep2 = (basep2 + confusionMatrix(as.factor(as.numeric(unlist(basepred2))), tes$class)$table) / 2
    basep3 = (basep3 + confusionMatrix(as.factor(as.numeric(unlist(basepred3))), tes$class)$table) / 2
    basep4 = (basep4 + confusionMatrix(as.factor(as.numeric(unlist(basepred4))), tes$class)$table) / 2
  
    selfp1 = (selfp1 + confusionMatrix(as.factor(as.numeric(unlist(selfpred1))), tes$class)$table) / 2
    selfp2 = (selfp2 + confusionMatrix(as.factor(as.numeric(unlist(selfpred2))), tes$class)$table) / 2
    selfp3 = (selfp3 + confusionMatrix(as.factor(as.numeric(unlist(selfpred3))), tes$class)$table) / 2
    selfp4 = (selfp4 + confusionMatrix(as.factor(as.numeric(unlist(selfpred4))), tes$class)$table) / 2
  
    baseauc1 = mean(baseauc1, multiclass.roc(tes$class,as.numeric(unlist(basepred1)))$auc[1])
    baseauc2 = mean(baseauc2, multiclass.roc(tes$class,as.numeric(unlist(basepred2)))$auc[1])
    baseauc3 = mean(baseauc3, multiclass.roc(tes$class,as.numeric(unlist(basepred3)))$auc[1])
    baseauc4 = mean(baseauc4, multiclass.roc(tes$class,as.numeric(unlist(basepred4)))$auc[1])
    
    selfauc1 = mean(selfauc1, multiclass.roc(tes$class,as.numeric(unlist(selfpred1)))$auc[1])
    selfauc2 = mean(selfauc2, multiclass.roc(tes$class,as.numeric(unlist(selfpred2)))$auc[1])
    selfauc3 = mean(selfauc3, multiclass.roc(tes$class,as.numeric(unlist(selfpred3)))$auc[1])
    selfauc4 = mean(selfauc4, multiclass.roc(tes$class,as.numeric(unlist(selfpred4)))$auc[1])
  }
}

#Clear up memory
rm(basefit1, basefit2, basefit3, basefit4, basepred1, basepred2, basepred3, basepred4, dectree,
   folds, knn, labelledData, logreg, selffit1, selffit2, selffit3, selffit4, selfpred1, selfpred2,
   selfpred3, selfpred4, semifolds, semitra, svmrbf, tes, tra, unlabelledData)
gc()

toc(log=TRUE, quiet = TRUE)
duration = capture.output(tic.log(format = TRUE))[2]
duration
