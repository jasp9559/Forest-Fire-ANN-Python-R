# Fire-Forests-Python-R

Fire Forests data wherein we try to predict the area of the forest fire spread based on the given parameters

  We have a dataset about 517 fires from the Montesano natural park in Portugal. For each incident, weekday, month, coordinates, and the burnt area are recorded, as well as several meteorological data such as rain, temperature, humidity, and wind. Predict the burnt area of forest fires with the help of an Artificial Neural Network model.

Solution in Python:
 
 a.	We load in the libraries required for the model development
 
 b.	We load in the forest fire dataset
 
 c.	We see that the data has uneven data with different scales, hence we use normalization for the data and normalize the data to get the values across within the range of 0 to 1.
 
 d.	Now we define a custom function wherein the hidden neurons are defined for neural network with kernel initializer as “normal” and activation function as “sigmoid” for I = 1, starting with I = 0 and building a dense network. Compile this with the loss function taken as “binary cross entropy” and optimizer as “rmsprop”, since the output variable is transformed to discrete and categorical = 2nos.
 
 e.	Split the data into train and test based on the selected predictors and target.
 
 f.	We now create a model with the defined function above and set with a input of 8 features and the hidden layers/ neurons as 50, 40, 20  and 1 at last.
 
 g.	This model is fit for working with 750 epochs, making it better to learn with more runs.
 
 h.	 Now predicting the data using the above for train and test gives the following results for the Correlation coefficients between the predicted values set as large or small and actual values
          
          i)	Prediction for train Cor – 98.54%
          ii)	Prediction for test Cor – 95.19%
 
 i.	We see that the model developed returns good values for correlation. Hence we can use the same to help classify the area of forest fire based on the parameters defined.
