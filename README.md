# PPI_pred
Required libraries and the terminal command to install the libraries:
-numpy: "sudo pip install numpy"
-sklearn: "sudo pip install sklearn"
-matplotlib.pyplot: "sudo apt-get install python-matplotlib"

Running the code:
Data preprocessing:
1. To get protein sequences, run python script using following command in terminal:
"python get_seq.py"
It will generate four outputs into "dataset" folder, including two protein sequence files for positive and negative samples, and two corresponding protein ID files for positive and  negative files, named "ara_seq", "neg_seq", "ara_id", and "neg_id" respectively.

2. To run Matlab scripts, open matlab and execute the script by the following command lines"
"run run_pos_feature.m"
"run run_neg_feature.m"
It will generate two output files with the sequence features for positive and negative samples into the "dataset" folder, named "ara_svm_input" and "neg_svm_input", respectively. These process will take four hours.

Run nested cross-validation and results
3. To run nested cross-validation scripts, run python script using following command in terminal:
"python nest_cv.py"
It will run the nested cross validation for the inner loop to select the best hyperparameters with the highest averaged accuracy in the inner loop, and using this combination to fit a model and make the prediction on the test set on the outer loop. It will generate five folders into "output" folder. Four "fold_n_acc" files contains the accuracy with each combination of parameters in the nth fold. "performance" contains the accuracy using the best hyperparameters in the nth outer loop testing set. Running this script will take around 3 to 4 hours. To run the smaller data set, please comment out the labeled lines in the "nested_cv.py" script (lines from 14 to 18 in the script) to test the code on the smaller data set including 100 positive samples and 100 negative samples.

4. To calculate the recall, precision, specificity, and F1 score, run python script using following command in terminal:
"python compute_score.py"
It will generate files named "scores" into the output folder. Running this will take around 10 minutes. 

5. To plot the ROC curve, run python script using following command in terminal:
"python roc_curve.py"
It will generate the roc curve and calculate the AUC score for each nth outer loop testing set into the output folder.

