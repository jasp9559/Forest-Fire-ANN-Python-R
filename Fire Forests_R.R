##### Neural Networks 
# Load the Fire Forests data

fire <- read.csv(file.choose())

# custom normalization function
norm <- function(x) {
  return((x - min(x))/ (max(x)- min(x)))
}

str(fire)
sum(is.na(fire))

View(fire)
summary(fire)

fire <- fire[ , -c(1, 2)]

# apply normalization to entire data frame
fire_norm <- as.data.frame(lapply(fire, norm))

# Now lets perform PCA over the data
pcaObj <- princomp(fire_norm, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:19]

# Top 19 pca scores making upto 90%
final <- cbind(fire_norm[, 9], pcaObj$scores[, 1:19])
View(final)
final <- as.data.frame(final)
final$fire <- final$V1
final <- final[ , c(21, c(2:20))]

#######________________################___________________##############
# create training and test data
fire_train <- final[1:465, ]
fire_test <- final[466:517, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
fire_model <- neuralnet(formula = fire ~ Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8 + Comp.9 + Comp.10 + Comp.11 + Comp.12 + Comp.13 + Comp.14 + Comp.15 + Comp.16 + Comp.17 + Comp.18 + Comp.19, data = fire_train)

# visualize the network topology
plot(fire_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(fire_model, fire_test[2:20])

# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, fire_test$fire)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
fire_model2 <- neuralnet(formula = fire ~ Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8 + Comp.9 + Comp.10 + Comp.11 + Comp.12 + Comp.13 + Comp.14 + Comp.15 + Comp.16 + Comp.17 + Comp.18 + Comp.19, data = fire_train, hidden = c(20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1))

# evaluate the results as we did before
model_results2 <- compute(fire_model2, fire_test[2:20])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, fire_test$fire)
