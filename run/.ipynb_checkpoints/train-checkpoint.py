


#Steps are: 

# Using a config input:

#Read data

#Preprocess (select variables of interest) + remove duplicates + nans

#Calculate weights ( + remove negative or 0 weights)

#Scale the input features and save the scalers

#Train test split (per sample), can be tied into making the datasets

#Read model (from config)

#Training loop
# - evalutate on validation stuff
# - add option for CV ? 

#Save outputs + trained model

#Test the model on new signal data
# - produce AD score
# - Save all relevant variables
# - Calculate separation from the Sig vs Bkg
# - need to separate the histogram production from the plotting

#Put the anomaly score back into the ROOT files
