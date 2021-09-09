#Takes ~12 hours

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
library(discrim)
library(rules)

#Record the duration of the experiment
tic.clearlog()
tic("Timer", msg = "Total time")

set.seed(1234)

#Read csv files
# data1M = read.csv(file="C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/dataset/data1M.csv", stringsAsFactors=FALSE)
data1M = read.csv(file="C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/metadata1M/metadata1M.csv", stringsAsFactors=FALSE)
meta = data1M$grade
data1M = data1M[,c(9,11)]

#Create corpus
corp = corpus(data1M$content)

#Attach class labels to the corpus as docvars
docvars(corp, "class") = paste0(data1M$class)

#Clean the text
data1M_dfm = tokens(corp, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)
data1M_dfm = tokens_remove(data1M_dfm, pattern = stopwords("en"))
data1M_dfm = dfm(data1M_dfm)
data1M_dfm = dfm_wordstem(data1M_dfm)

#Trim features by term frequency, document frequency, feature character count
data1M_dfm = dfm_trim(data1M_dfm, min_termfreq = 5, min_docfreq = 10000)
data1M_dfm = dfm_select(data1M_dfm, min_nchar = 3)

#Apply tf-idf to document feature matrix
data1M_dfm = dfm_tfidf(data1M_dfm, scheme_tf = "count", scheme_df = "inverse")

#Convert document feature matrix to data frame
data1M_dfm = cbind(convert(data1M_dfm, to = "data.frame"), docvars(data1M_dfm))

#Remove the document column that has been added by default
data1M_dfm$document = NULL

# #Add metadata
# data1M_dfm = cbind(data1M_dfm, meta)

#Split the data into labelled and unlabelled, randomly select required amounts
labelledData = data1M_dfm[1:2757,]
unlabelledData = data1M_dfm[2758:nrow(data1M_dfm),]

#Remove objects and free up some memory
rm(data1M, corp, data1M_dfm, meta)
gc()

########## Create training and testing data section ##########
#Splitting our labelled dataset 2757 into 2, for training and for algorithm performance dataset

#Reduce data to smaller set
labelledData = stratified(labelledData, 'class', 0.5)
unlabelledData = unlabelledData[sample(nrow(unlabelledData), 6895),]

#Create folds with stratified sampling
folds = createFolds(factor(labelledData$class), k = 10, list = T)
semifolds = createFolds(unlabelledData$class, k = 50, list = T)
for (i in 2:50){
  semifolds[[i]] = c(semifolds[[i]], semifolds[[i-1]])
}

#Designate base classifiers
c5 = C5_rules(mode = "classification") %>% set_engine("C5.0")
knn = nearest_neighbor(mode = "classification") %>% set_engine("kknn")
logreg = multinom_reg(mode = "classification") %>% set_engine("nnet")

# Change classes to number
labelledData$class = as.factor(as.numeric(as.factor(labelledData$class)))
unlabelledData$class = as.factor(rep(NA, nrow(unlabelledData)))

#Initialize performance metric lists
p1 = vector("list", 51)
p2 = vector("list", 51)
p3 = vector("list", 51)

auc1 = vector("list", 51)
auc2 = vector("list", 51)
auc3 = vector("list", 51)

for (i in 10:10){
  #Define training and testing sets for base classifiers
  tes = labelledData[folds[[i]],]
  tra = labelledData[-folds[[i]],]
  
  #Train using base classifiers
  basefit1 = fit(object = c5, class~., data = tra[,-1])
  basefit2 = fit(object = knn, class~., data = tra[,-1])
  basefit3 = fit(object = logreg, class~., data = tra[,-1])
  
  #Predict using base classifiers
  basepred1 = predict(basefit1, tes[,-1])
  basepred2 = predict(basefit2, tes[,-1])
  basepred3 = predict(basefit3, tes[,-1])
  
  #Record confusion matrices and auc values
  if (i == 1){
    p1[[1]] = confusionMatrix(as.factor(as.numeric(unlist(basepred1))), tes$class)$table
    p2[[1]] = confusionMatrix(as.factor(as.numeric(unlist(basepred2))), tes$class)$table
    p3[[1]] = confusionMatrix(as.factor(as.numeric(unlist(basepred3))), tes$class)$table
    
    auc1[1] = multiclass.roc(tes$class, as.numeric(unlist(basepred1)))$auc[1]
    auc2[1] = multiclass.roc(tes$class, as.numeric(unlist(basepred2)))$auc[1]
    auc3[1] = multiclass.roc(tes$class, as.numeric(unlist(basepred3)))$auc[1]
    
  } else{
    p1[[1]] = (p1[[1]] + confusionMatrix(as.factor(as.numeric(unlist(basepred1))), tes$class)$table) / 2
    p2[[1]] = (p2[[1]] + confusionMatrix(as.factor(as.numeric(unlist(basepred2))), tes$class)$table) / 2
    p3[[1]] = (p3[[1]] + confusionMatrix(as.factor(as.numeric(unlist(basepred3))), tes$class)$table) / 2
    
    auc1[1] = mean(auc1[[1]], multiclass.roc(tes$class,as.numeric(unlist(basepred1)))$auc[1])
    auc2[1] = mean(auc2[[1]], multiclass.roc(tes$class,as.numeric(unlist(basepred2)))$auc[1])
    auc3[1] = mean(auc3[[1]], multiclass.roc(tes$class,as.numeric(unlist(basepred3)))$auc[1])
  }
  
  for (j in 2:51){
    #Define training set for self training
    tra = rbind(labelledData[-folds[[i]],], unlabelledData[semifolds[[j - 1]],])

    #Train wrapper algorithms
    selffit1 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = tra[,-1])
    selffit2 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = tra[,-1])
    selffit3 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class ~., data = tra[,-1])

    # Make predictions with wrapper algorithm
    selfpred1 = predict(selffit1, tes[,-1])
    selfpred2 = predict(selffit2, tes[,-1])
    selfpred3 = predict(selffit3, tes[,-1])

    #Record confusion matrices and auc values
    if (i == 1){
      p1[[j]] = confusionMatrix(as.factor(as.numeric(unlist(selfpred1))), tes$class)$table
      p2[[j]] = confusionMatrix(as.factor(as.numeric(unlist(selfpred2))), tes$class)$table
      p3[[j]] = confusionMatrix(as.factor(as.numeric(unlist(selfpred3))), tes$class)$table

      auc1[j] = multiclass.roc(tes$class,as.numeric(unlist(selfpred1)))$auc[1]
      auc2[j] = multiclass.roc(tes$class,as.numeric(unlist(selfpred2)))$auc[1]
      auc3[j] = multiclass.roc(tes$class,as.numeric(unlist(selfpred3)))$auc[1]

    } else{
      p1[[j]] = (p1[[j]] + confusionMatrix(as.factor(as.numeric(unlist(selfpred1))), tes$class)$table) / 2
      p2[[j]] = (p2[[j]] + confusionMatrix(as.factor(as.numeric(unlist(selfpred2))), tes$class)$table) / 2
      p3[[j]] = (p3[[j]] + confusionMatrix(as.factor(as.numeric(unlist(selfpred3))), tes$class)$table) / 2

      auc1[j] = mean(auc1[[j]], multiclass.roc(tes$class,as.numeric(unlist(selfpred1)))$auc[1])
      auc2[j] = mean(auc2[[j]], multiclass.roc(tes$class,as.numeric(unlist(selfpred2)))$auc[1])
      auc3[j] = mean(auc3[[j]], multiclass.roc(tes$class,as.numeric(unlist(selfpred3)))$auc[1])
    }

  }
  message(i)
}

#Clear up memory
rm(basefit1, basefit2, basefit3, basepred1, basepred2, basepred3, c5,
   folds, knn, labelledData, logreg, selffit1, selffit2, selffit3, selfpred1, selfpred2,
   selfpred3, semifolds, tes, tra, unlabelledData, i, j)
gc()

#Store results and plot
auc1 = sapply(auc1, mean)
auc2 = sapply(auc2, mean)
auc3 = sapply(auc3, mean)

aucdf = data.frame("Model" = c(rep("C5 Rules", 51), rep("kNN", 51), rep("Logistic Regression", 51)),
                   "% of unlabelled data" = rep(seq(0,3,0.1), 3), "auc" = c(auc1, auc2, auc3))

ggplot(data = aucdf) + geom_line(aes(y = auc, x = X..of.unlabelled.data, color = Model))

#Store time taken
toc(log=TRUE, quiet = TRUE)
duration = capture.output(tic.log(format = TRUE))[2]
duration
