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

fire <- fire[ , -c(1, 2, 12:30)]

fire$area <- ifelse(fire$area > 50, 1, 0)

# apply normalization to entire data frame
fire_norm <- as.data.frame(lapply(fire[ , c(1:8)], norm))
fire_norm <- cbind(fire$area, fire_norm)


#######________________################___________________##############
# create training and test data
fire_train <- fire_norm[1:415, ]
fire_test <- fire_norm[416:517, ]
attach(fire_train)
## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
fire_model <- neuralnet(formula = `fire$area` ~ FFMC + DMC + DC + ISI + temp + RH + wind + rain, data = fire_train)

# visualize the network topology
plot(fire_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(fire_model, fire_test[2:9])

# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, fire_test$`fire$area`)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
fire_model2 <- neuralnet(formula = `fire$area` ~ FFMC + DMC + DC + ISI + temp + RH + wind + rain, data = fire_train, hidden = c(12, 10, 8, 6, 4, 2, 1))


# evaluate the results as we did before
model_results2 <- compute(fire_model2, fire_test[2:9])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, fire_test$`fire$area`)
