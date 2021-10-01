#Takes ~15 hours on my machine

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
library(discrim)
library(rules)
library(fastDummies)
library(mltools)

#Record the duration of the experiment
timeTotal = Sys.time()
timePre = Sys.time()

#Set seed for reproducible results
set.seed(12345)

########## Pre-processing ##########

#Read csv files and isolate metadata
data1M = read.csv(file="C:/Users/tobyb/Documents/Uni_Links/Dissertation/AppStore/Dataset1M_2018/metadata1M/metadata1M.csv", stringsAsFactors=FALSE)
rating = data1M$grade
len = sapply(data1M$content, nchar)
price = data1M$price
data1M = dummy_cols(data1M, select_columns = 'category')
categories = data1M[,12:22]
data1M = data1M[,c(9,11)]

#Create corpus
corp = corpus(data1M$content)

#Attach class labels to the corpus
docvars(corp, "class") = paste0(data1M$class)

#Clean and transform into a document feature matrix
data1M_dfm = tokens(corp, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)
data1M_dfm = tokens_remove(data1M_dfm, pattern = stopwords("en"))
data1M_dfm = dfm(data1M_dfm)
data1M_dfm = dfm_wordstem(data1M_dfm)

#Trim features by term frequency, document frequency, feature character count
data1M_dfm = dfm_trim(data1M_dfm, min_termfreq = 5, min_docfreq = 10000)
data1M_dfm = dfm_select(data1M_dfm, min_nchar = 3)

#Apply tf-idf
data1M_dfm = dfm_tfidf(data1M_dfm, scheme_tf = "count", scheme_df = "inverse")

#Convert document feature matrix to data frame
data1M_dfm = cbind(convert(data1M_dfm, to = "data.frame"), docvars(data1M_dfm))

#Remove the document and doc_id columns
data1M_dfm$document = NULL
data1M_dfm$doc_id = NULL

########## Add metadata as appropriate ##########
# data1M_dfm = cbind(data1M_dfm, rating)
# data1M_dfm = cbind(data1M_dfm, len)
# data1M_dfm = cbind(data1M_dfm, price)
# data1M_dfm = cbind(data1M_dfm, categories)

#Split the data into labelled and unlabelled
labelledData = data1M_dfm[1:2757,]
unlabelledData = data1M_dfm[2758:nrow(data1M_dfm),]

#Remove objects and free up some memory
rm(data1M, corp, data1M_dfm, rating, len, price, categories)
gc()

#Pre-processing time
timePre = Sys.time() - timePre

########## Prepare data for training and testing ##########

#Designate base classifiers
c5 = C5_rules(mode = "classification") %>% set_engine("C5.0")
knn = nearest_neighbor(mode = "classification") %>% set_engine("kknn")
logreg = multinom_reg(mode = "classification") %>% set_engine("nnet")

#Change classes to number
labelledData$class = as.factor(as.numeric(as.factor(labelledData$class)))
unlabelledData$class = as.factor(rep(NA, nrow(unlabelledData)))

#Create data frame to store results
df = data.frame("Model" = c(rep("C5.0 Decision Tree", 10), rep("k-NN", 10), rep("Multinomial Logistic Regression", 10)), "Percentage Labelled" = rep(c(10,20,30,40,50,60,70,80,90,100), 3), "MCC" = rep(0,30))

#Create lists to store 
c5p = vector("list", 10)
knnp = vector("list", 10)
logp = vector("list", 10)

########## Run experiment ##########

timeExperiment = Sys.time()

classacc = function(confMatrix){
  acc = c()
  for (y in 1:3){
    if (sum(confMatrix[,y]) == 0){
      acc = c(acc, NA)
    } else{
    acc = c(acc, confMatrix[y,y]/sum(confMatrix[,y]))
    }
  }
  return(acc)
}

for (j in 1:3){
  #Reduce size of unlabelled data and shuffle both
  unlabelledData = unlabelledData[sample(nrow(unlabelledData), 2757),]
  labelledData = labelledData[sample(nrow(labelledData), 2757),]
  
  #Create subsets to add to labelled data
  semi10 = stratified(unlabelledData, "class", 0.9)
  semi20 = stratified(unlabelledData, "class", 0.8)
  semi30 = stratified(unlabelledData, "class", 0.7)
  semi40 = stratified(unlabelledData, "class", 0.6)
  semi50 = stratified(unlabelledData, "class", 0.5)
  semi60 = stratified(unlabelledData, "class", 0.4)
  semi70 = stratified(unlabelledData, "class", 0.3)
  semi80 = stratified(unlabelledData, "class", 0.2)
  semi90 = stratified(unlabelledData, "class", 0.1)
  
  #Create folds for training base classifiers and testing all data
  folds = createFolds(factor(labelledData$class), k = 10, list = T)
  
  #Create folds for cross validation
  semifolds10 = createFolds(1:nrow(semi10), k = 10, list = T)
  semifolds20 = createFolds(1:nrow(semi20), k = 10, list = T)
  semifolds30 = createFolds(1:nrow(semi30), k = 10, list = T)
  semifolds40 = createFolds(1:nrow(semi40), k = 10, list = T)
  semifolds50 = createFolds(1:nrow(semi50), k = 10, list = T)
  semifolds60 = createFolds(1:nrow(semi60), k = 10, list = T)
  semifolds70 = createFolds(1:nrow(semi70), k = 10, list = T)
  semifolds80 = createFolds(1:nrow(semi80), k = 10, list = T)
  semifolds90 = createFolds(1:nrow(semi90), k = 10, list = T)
  
  #Perform 10 fold cross validation
  for (i in 1:10){
    
    #Create training sets
    train10 = stratified(labelledData[-folds[[i]],], "class", 0.1)
    train20 = stratified(labelledData[-folds[[i]],], "class", 0.2)
    train30 = stratified(labelledData[-folds[[i]],], "class", 0.3)
    train40 = stratified(labelledData[-folds[[i]],], "class", 0.4)
    train50 = stratified(labelledData[-folds[[i]],], "class", 0.5)
    train60 = stratified(labelledData[-folds[[i]],], "class", 0.6)
    train70 = stratified(labelledData[-folds[[i]],], "class", 0.7)
    train80 = stratified(labelledData[-folds[[i]],], "class", 0.8)
    train90 = stratified(labelledData[-folds[[i]],], "class", 0.9)
    
    #Train algorithms
    c5fit10 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train10, unlabelledData[do.call(c, semifolds10[-i]),]))
    c5fit20 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train20, unlabelledData[do.call(c, semifolds20[-i]),]))
    c5fit30 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train30, unlabelledData[do.call(c, semifolds30[-i]),]))
    c5fit40 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train40, unlabelledData[do.call(c, semifolds40[-i]),]))
    c5fit50 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train50, unlabelledData[do.call(c, semifolds50[-i]),]))
    c5fit60 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train60, unlabelledData[do.call(c, semifolds60[-i]),]))
    c5fit70 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train70, unlabelledData[do.call(c, semifolds70[-i]),]))
    c5fit80 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train80, unlabelledData[do.call(c, semifolds80[-i]),]))
    c5fit90 = selfTraining(learner = c5, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train90, unlabelledData[do.call(c, semifolds90[-i]),]))
    c5fit100 = fit(object = c5, class~., data = labelledData[do.call(c, folds[-i]),])
    
    knnfit10 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train10, unlabelledData[do.call(c, semifolds10[-i]),]))
    knnfit20 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train20, unlabelledData[do.call(c, semifolds20[-i]),]))
    knnfit30 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train30, unlabelledData[do.call(c, semifolds30[-i]),]))
    knnfit40 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train40, unlabelledData[do.call(c, semifolds40[-i]),]))
    knnfit50 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train50, unlabelledData[do.call(c, semifolds50[-i]),]))
    knnfit60 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train60, unlabelledData[do.call(c, semifolds60[-i]),]))
    knnfit70 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train70, unlabelledData[do.call(c, semifolds70[-i]),]))
    knnfit80 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train80, unlabelledData[do.call(c, semifolds80[-i]),]))
    knnfit90 = selfTraining(learner = knn, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train90, unlabelledData[do.call(c, semifolds90[-i]),]))
    knnfit100 = fit(object = knn, class~., data = labelledData[do.call(c, folds[-i]),])
    
    logfit10 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train10, unlabelledData[do.call(c, semifolds10[-i]),]))
    logfit20 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train20, unlabelledData[do.call(c, semifolds20[-i]),]))
    logfit30 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train30, unlabelledData[do.call(c, semifolds30[-i]),]))
    logfit40 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train40, unlabelledData[do.call(c, semifolds40[-i]),]))
    logfit50 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train50, unlabelledData[do.call(c, semifolds50[-i]),]))
    logfit60 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train60, unlabelledData[do.call(c, semifolds60[-i]),]))
    logfit70 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train70, unlabelledData[do.call(c, semifolds70[-i]),]))
    logfit80 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train80, unlabelledData[do.call(c, semifolds80[-i]),]))
    logfit90 = selfTraining(learner = logreg, perc.full = 0.7, thr.conf = 0.5, max.iter = 50) %>% fit(class~., data = rbind(train90, unlabelledData[do.call(c, semifolds90[-i]),]))
    logfit100 = fit(object = logreg, class~., data = labelledData[do.call(c, folds[-i]),])
    
    #Make predictions on test data
    c5pred10 = predict(c5fit10, labelledData[folds[[i]],])
    c5pred20 = predict(c5fit20, labelledData[folds[[i]],])
    c5pred30 = predict(c5fit30, labelledData[folds[[i]],])
    c5pred40 = predict(c5fit40, labelledData[folds[[i]],])
    c5pred50 = predict(c5fit50, labelledData[folds[[i]],])
    c5pred60 = predict(c5fit60, labelledData[folds[[i]],])
    c5pred70 = predict(c5fit70, labelledData[folds[[i]],])
    c5pred80 = predict(c5fit80, labelledData[folds[[i]],])
    c5pred90 = predict(c5fit90, labelledData[folds[[i]],])
    c5pred100 = predict(c5fit100, labelledData[folds[[i]],])
    
    knnpred10 = predict(knnfit10, labelledData[folds[[i]],])
    knnpred20 = predict(knnfit20, labelledData[folds[[i]],])
    knnpred30 = predict(knnfit30, labelledData[folds[[i]],])
    knnpred40 = predict(knnfit40, labelledData[folds[[i]],])
    knnpred50 = predict(knnfit50, labelledData[folds[[i]],])
    knnpred60 = predict(knnfit60, labelledData[folds[[i]],])
    knnpred70 = predict(knnfit70, labelledData[folds[[i]],])
    knnpred80 = predict(knnfit80, labelledData[folds[[i]],])
    knnpred90 = predict(knnfit90, labelledData[folds[[i]],])
    knnpred100 = predict(knnfit100, labelledData[folds[[i]],])
    
    logpred10 = predict(logfit10, labelledData[folds[[i]],])
    logpred20 = predict(logfit20, labelledData[folds[[i]],])
    logpred30 = predict(logfit30, labelledData[folds[[i]],])
    logpred40 = predict(logfit40, labelledData[folds[[i]],])
    logpred50 = predict(logfit50, labelledData[folds[[i]],])
    logpred60 = predict(logfit60, labelledData[folds[[i]],])
    logpred70 = predict(logfit70, labelledData[folds[[i]],])
    logpred80 = predict(logfit80, labelledData[folds[[i]],])
    logpred90 = predict(logfit90, labelledData[folds[[i]],])
    logpred100 = predict(logfit100, labelledData[folds[[i]],])
    
    #Collect performance metrics
    if (i==1 & j==1){
      c5p[[1]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred10))), labelledData$class[folds[[i]]])$table)
      c5p[[2]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred20))), labelledData$class[folds[[i]]])$table)
      c5p[[3]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred30))), labelledData$class[folds[[i]]])$table)
      c5p[[4]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred40))), labelledData$class[folds[[i]]])$table)
      c5p[[5]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred50))), labelledData$class[folds[[i]]])$table)
      c5p[[6]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred60))), labelledData$class[folds[[i]]])$table)
      c5p[[7]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred70))), labelledData$class[folds[[i]]])$table)
      c5p[[8]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred80))), labelledData$class[folds[[i]]])$table)
      c5p[[9]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred90))), labelledData$class[folds[[i]]])$table)
      c5p[[10]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred100))), labelledData$class[folds[[i]]])$table)
      
      knnp[[1]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred10))), labelledData$class[folds[[i]]])$table)
      knnp[[2]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred20))), labelledData$class[folds[[i]]])$table)
      knnp[[3]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred30))), labelledData$class[folds[[i]]])$table)
      knnp[[4]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred40))), labelledData$class[folds[[i]]])$table)
      knnp[[5]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred50))), labelledData$class[folds[[i]]])$table)
      knnp[[6]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred60))), labelledData$class[folds[[i]]])$table)
      knnp[[7]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred70))), labelledData$class[folds[[i]]])$table)
      knnp[[8]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred80))), labelledData$class[folds[[i]]])$table)
      knnp[[9]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred90))), labelledData$class[folds[[i]]])$table)
      knnp[[10]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred100))), labelledData$class[folds[[i]]])$table)
      
      logp[[1]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred10))), labelledData$class[folds[[i]]])$table)
      logp[[2]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred20))), labelledData$class[folds[[i]]])$table)
      logp[[3]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred30))), labelledData$class[folds[[i]]])$table)
      logp[[4]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred40))), labelledData$class[folds[[i]]])$table)
      logp[[5]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred50))), labelledData$class[folds[[i]]])$table)
      logp[[6]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred60))), labelledData$class[folds[[i]]])$table)
      logp[[7]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred70))), labelledData$class[folds[[i]]])$table)
      logp[[8]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred80))), labelledData$class[folds[[i]]])$table)
      logp[[9]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred90))), labelledData$class[folds[[i]]])$table)
      logp[[10]] = classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred100))), labelledData$class[folds[[i]]])$table)
      
      df$MCC[1] = mcc(unlist(c5pred10), labelledData$class[folds[[i]]])
      df$MCC[2] = mcc(unlist(c5pred20), labelledData$class[folds[[i]]])
      df$MCC[3] = mcc(unlist(c5pred30), labelledData$class[folds[[i]]])
      df$MCC[4] = mcc(unlist(c5pred40), labelledData$class[folds[[i]]])
      df$MCC[5] = mcc(unlist(c5pred50), labelledData$class[folds[[i]]])
      df$MCC[6] = mcc(unlist(c5pred60), labelledData$class[folds[[i]]])
      df$MCC[7] = mcc(unlist(c5pred70), labelledData$class[folds[[i]]])
      df$MCC[8] = mcc(unlist(c5pred80), labelledData$class[folds[[i]]])
      df$MCC[9] = mcc(unlist(c5pred90), labelledData$class[folds[[i]]])
      df$MCC[10] = mcc(unlist(c5pred100), labelledData$class[folds[[i]]])
      
      df$MCC[11] = mcc(unlist(knnpred10), labelledData$class[folds[[i]]])
      df$MCC[12] = mcc(unlist(knnpred20), labelledData$class[folds[[i]]])
      df$MCC[13] = mcc(unlist(knnpred30), labelledData$class[folds[[i]]])
      df$MCC[14] = mcc(unlist(knnpred40), labelledData$class[folds[[i]]])
      df$MCC[15] = mcc(unlist(knnpred50), labelledData$class[folds[[i]]])
      df$MCC[16] = mcc(unlist(knnpred60), labelledData$class[folds[[i]]])
      df$MCC[17] = mcc(unlist(knnpred70), labelledData$class[folds[[i]]])
      df$MCC[18] = mcc(unlist(knnpred80), labelledData$class[folds[[i]]])
      df$MCC[19] = mcc(unlist(knnpred90), labelledData$class[folds[[i]]])
      df$MCC[20] = mcc(unlist(knnpred100), labelledData$class[folds[[i]]])
      
      df$MCC[21] = mcc(unlist(logpred10), labelledData$class[folds[[i]]])
      df$MCC[22] = mcc(unlist(logpred20), labelledData$class[folds[[i]]])
      df$MCC[23] = mcc(unlist(logpred30), labelledData$class[folds[[i]]])
      df$MCC[24] = mcc(unlist(logpred40), labelledData$class[folds[[i]]])
      df$MCC[25] = mcc(unlist(logpred50), labelledData$class[folds[[i]]])
      df$MCC[26] = mcc(unlist(logpred60), labelledData$class[folds[[i]]])
      df$MCC[27] = mcc(unlist(logpred70), labelledData$class[folds[[i]]])
      df$MCC[28] = mcc(unlist(logpred80), labelledData$class[folds[[i]]])
      df$MCC[29] = mcc(unlist(logpred90), labelledData$class[folds[[i]]])
      df$MCC[30] = mcc(unlist(logpred100), labelledData$class[folds[[i]]])
      
    } else{
      c5p[[1]] = mean(c5p[[1]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred10))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[2]] mean(c5p[[2]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred20))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[3]] mean(c5p[[3]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred30))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[4]] mean(c5p[[4]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred40))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[5]] mean(c5p[[5]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred50))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[6]] mean(c5p[[6]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred60))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[7]] mean(c5p[[7]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred70))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[8]] mean(c5p[[8]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred80))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[9]] mean(c5p[[9]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred90))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      c5p[[10]] mean(c5p[[10]], classacc(confusionMatrix(as.factor(as.numeric(unlist(c5pred100))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      
      knnp[[1]] mean(knnp[[1]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred10))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[2]] mean(knnp[[2]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred20))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[3]] mean(knnp[[3]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred30))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[4]] mean(knnp[[4]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred40))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[5]] mean(knnp[[5]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred50))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[6]] mean(knnp[[6]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred60))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[7]] mean(knnp[[7]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred70))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[8]] mean(knnp[[8]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred80))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[9]] mean(knnp[[9]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred90))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      knnp[[10]] mean(knnp[[10]], classacc(confusionMatrix(as.factor(as.numeric(unlist(knnpred100))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      
      logp[[1]] mean(logp[[1]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred10))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[2]] mean(logp[[2]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred20))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[3]] mean(logp[[3]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred30))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[4]] mean(logp[[4]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred40))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[5]] mean(logp[[5]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred50))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[6]] mean(logp[[6]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred60))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[7]] mean(logp[[7]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred70))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[8]] mean(logp[[8]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred80))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[9]] mean(logp[[9]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred90))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
      logp[[10]] mean(logp[[10]], classacc(confusionMatrix(as.factor(as.numeric(unlist(logpred100))), labelledData$class[folds[[i]]])$table), na.rm = TRUE)
     
      df$MCC[1] = mean(df$MCC[1], mcc(unlist(c5pred10), labelledData$class[folds[[i]]]))
      df$MCC[2] = mean(df$MCC[2], mcc(unlist(c5pred20), labelledData$class[folds[[i]]]))
      df$MCC[3] = mean(df$MCC[3], mcc(unlist(c5pred30), labelledData$class[folds[[i]]]))
      df$MCC[4] = mean(df$MCC[4], mcc(unlist(c5pred40), labelledData$class[folds[[i]]]))
      df$MCC[5] = mean(df$MCC[5], mcc(unlist(c5pred50), labelledData$class[folds[[i]]]))
      df$MCC[6] = mean(df$MCC[6], mcc(unlist(c5pred60), labelledData$class[folds[[i]]]))
      df$MCC[7] = mean(df$MCC[7], mcc(unlist(c5pred70), labelledData$class[folds[[i]]]))
      df$MCC[8] = mean(df$MCC[8], mcc(unlist(c5pred80), labelledData$class[folds[[i]]]))
      df$MCC[9] = mean(df$MCC[9], mcc(unlist(c5pred90), labelledData$class[folds[[i]]]))
      df$MCC[10] = mean(df$MCC[10], mcc(unlist(c5pred100), labelledData$class[folds[[i]]]))
      
      df$MCC[11] = mean(df$MCC[11], mcc(unlist(knnpred10), labelledData$class[folds[[i]]]))
      df$MCC[12] = mean(df$MCC[12], mcc(unlist(knnpred20), labelledData$class[folds[[i]]]))
      df$MCC[13] = mean(df$MCC[13], mcc(unlist(knnpred30), labelledData$class[folds[[i]]]))
      df$MCC[14] = mean(df$MCC[14], mcc(unlist(knnpred40), labelledData$class[folds[[i]]]))
      df$MCC[15] = mean(df$MCC[15], mcc(unlist(knnpred50), labelledData$class[folds[[i]]]))
      df$MCC[16] = mean(df$MCC[16], mcc(unlist(knnpred60), labelledData$class[folds[[i]]]))
      df$MCC[17] = mean(df$MCC[17], mcc(unlist(knnpred70), labelledData$class[folds[[i]]]))
      df$MCC[18] = mean(df$MCC[18], mcc(unlist(knnpred80), labelledData$class[folds[[i]]]))
      df$MCC[19] = mean(df$MCC[19], mcc(unlist(knnpred90), labelledData$class[folds[[i]]]))
      df$MCC[20] = mean(df$MCC[20], mcc(unlist(knnpred100), labelledData$class[folds[[i]]]))
      
      df$MCC[21] = mean(df$MCC[21], mcc(unlist(logpred10), labelledData$class[folds[[i]]]))
      df$MCC[22] = mean(df$MCC[22], mcc(unlist(logpred20), labelledData$class[folds[[i]]]))
      df$MCC[23] = mean(df$MCC[23], mcc(unlist(logpred30), labelledData$class[folds[[i]]]))
      df$MCC[24] = mean(df$MCC[24], mcc(unlist(logpred40), labelledData$class[folds[[i]]]))
      df$MCC[25] = mean(df$MCC[25], mcc(unlist(logpred50), labelledData$class[folds[[i]]]))
      df$MCC[26] = mean(df$MCC[26], mcc(unlist(logpred60), labelledData$class[folds[[i]]]))
      df$MCC[27] = mean(df$MCC[27], mcc(unlist(logpred70), labelledData$class[folds[[i]]]))
      df$MCC[28] = mean(df$MCC[28], mcc(unlist(logpred80), labelledData$class[folds[[i]]]))
      df$MCC[29] = mean(df$MCC[29], mcc(unlist(logpred90), labelledData$class[folds[[i]]]))
      df$MCC[30] = mean(df$MCC[30], mcc(unlist(logpred100), labelledData$class[folds[[i]]]))
    }
    message(paste0(round(100*(10*(j-1)+i)/30, 2), "% Done, ", "Time running: ", round(Sys.time() - timeTotal, 2)))
  }
}

rm(list = ls()[! ls() %in% c("df", "duration", "c5p", "knnp", "logp", "basedfauc", "lengthdfauc", "ratingdfauc", "catdfauc", "pricedfauc", "timeTotal", "timePre", "timeExperiment", "dftimes")])
gc()

########## Store results in appropriate dataframe ##########
# basedfauc = df
# ratingdfauc = df
# lengthdfauc = df
# pricedfauc = df
catdfauc = df

#Store time taken
timeTotal = Sys.time() - timeTotal
timeExperiment = Sys.time() - timeExperiment
