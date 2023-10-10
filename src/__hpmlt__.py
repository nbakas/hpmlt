

#############################################################################
# More instructions in README.md and __hpmlt__.pdf
#############################################################################


#############################################################################
# Define the problem
# Simply define the main parameters here. The code will automatically produce the corresponding graphs and tables.
ROOT_DIR = "/.../.../"
# A *****ROOT_DIR without the filename***** with only one excel file.
# The *.xlsx file shhould comprise with all the independent variables at the first $n$ columns, followed by the target variable as the last column. 
# For Windows, please use *****\\***** separators and remeber to *****add the \\ at the end*****. 
# For Linux please use /.../.../ format
#############################################################################
LOGISTIC_REGR = False # If True do classification
#############################################################################
PERMUTE_TRAIN_TEST = True # If True split the data into training/testing sets randomly, after shuffling. 
# If False, Top rows are train and bottom test, which is helpfull for time series data.
#############################################################################
IS_HPC = True #True only on HPC with export OMP_NUM_THREADS=1
# the total number of distributed threads should not exceed cpu_count, if blas_threads>1
#############################################################################


#############################################################################
# Import Libraries
from import_libraries import *
import import_libraries, misc_functions, descriptive_statistics, ml_linear_regression, ml_nlregr
import ml_xgboost, ml_ANNBN, ml_random_forests, ml_DANN
reload(import_libraries); reload(misc_functions); reload(descriptive_statistics); reload(ml_linear_regression); reload(ml_nlregr)
reload(ml_xgboost); reload(ml_ANNBN); reload(ml_random_forests); reload(ml_DANN)
#############################################################################


#############################################################################
# misc_functions.delete_files_except_xlsx([ROOT_DIR])
#############################################################################
# Open the Dataset and Split to Train & Test
test_ratio = 0.5 #The ratio of the data to be used for testing (0-1). Usually 0.2-0.3
random_seed = 0 #The random seed to be used for the random number generator.
df, features_names, target_name = misc_functions.read_the_dataset_dropna(ROOT_DIR)
df, features_names, target_name = misc_functions.make_features_and_target_categorical_if_so(df, target_name, ROOT_DIR, True)
Xtr, Xte, ytr, yte = misc_functions.split_train_test(PERMUTE_TRAIN_TEST, test_ratio, random_seed, df)
misc_functions.create_all_directories(ROOT_DIR, PERMUTE_TRAIN_TEST)
#############################################################################


#############################################################################
# Clean the Data
# Xte,yte = misc_functions.delete_missing_and_inf_values_rows(Xte,yte)
# Xtr,ytr = misc_functions.delete_missing_and_inf_values_rows(Xtr,ytr)
# Xtr,ytr = misc_functions.delete_identical_rows(Xtr,ytr)
# Xtr, Xte, features_names = misc_functions.check_fix_multicolinearity(Xtr, Xte, features_names, ROOT_DIR)
#############################################################################


#############################################################################
# Descriptive Statistics 
# descriptive_statistics.descriptive_statistics(Xtr, ytr, Xte, yte, features_names, target_name, ROOT_DIR)
# descriptive_statistics.plot_pdf_cdf_all(Xtr, ytr, features_names, target_name, ROOT_DIR, "_Train")
# descriptive_statistics.plot_pdf_cdf_all(Xte, yte, features_names, target_name, ROOT_DIR, "_Test")
# descriptive_statistics.plot_all_by_all_correlation_matrix(Xtr, ytr, features_names, target_name, ROOT_DIR)
#############################################################################


#############################################################################
# Machine Learning
#############################################################################

## Linear Regression
# ml_linear_regression.do_regression(Xtr, Xte, ytr, yte, features_names, target_name, ROOT_DIR, LOGISTIC_REGR)

## Polynomial Regression
# ml_nlregr.do_nlregr(Xtr, Xte, ytr, yte, features_names, target_name, LOGISTIC_REGR, PERMUTE_TRAIN_TEST, ROOT_DIR)

## XGBoost
nof_1st_tune_rounds = 100; nof_1st_tune_models = 100
nof_2nd_tune_rounds = 1000; nof_2nd_tune_models = 10
nof_final_blas_thr = 5
ml_xgboost.do_xgboost(Xtr,Xte,ytr,yte,features_names,target_name,IS_HPC,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR,
                      nof_1st_tune_rounds,nof_1st_tune_models,nof_2nd_tune_rounds,nof_2nd_tune_models,
                      nof_final_blas_thr)

## Random Forests
# __thres_early_stop__ = 1e-2
# __thres_min_tune_rounds__ = 100
# ml_random_forests.do_random_forests(Xtr,Xte,ytr,yte,features_names,target_name,__thres_early_stop__,
#                                     __thres_min_tune_rounds__,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR)

## ANNBN
# max_nof_aug_samples = 10_000; noise_factor = 0.2
# Xtr_all, ytr_all = ml_ANNBN.augment_train_dataset(Xtr,ytr,max_nof_aug_samples,noise_factor)
# neurons_all = [int(lr1) for lr1 in linspace(len(ytr_all)/40,len(ytr_all)/10,num=5)]
# ml_ANNBN.do_ANNBN(Xtr_all, Xte, ytr_all, yte, features_names, target_name, neurons_all,
#                   PERMUTE_TRAIN_TEST, LOGISTIC_REGR, ROOT_DIR)

## Deep Learning
nof_1st_tune_epochs = 10; nof_1st_tune_models = 20
nof_2nd_tune_epochs = 100; nof_2nd_tune_models = 4
nof_final_blas_thr = 5
ml_DANN.do_DANN(Xtr,ytr,Xte,yte,features_names,target_name,IS_HPC,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR,
                nof_1st_tune_models,nof_1st_tune_epochs,nof_2nd_tune_models,nof_2nd_tune_epochs,nof_final_blas_thr)


#############################################################################
# Sensitivity Analysis
# sens_methods = ["LinRegr", "NLRegr", "XGBoost", "RF", "ANNBN", "DANN"]
sens_methods = ["XGBoost", "DANN"]
misc_functions.gather_all_sensitivity_curves(Xtr, ytr, features_names, target_name, sens_methods, ROOT_DIR)
#############################################################################


#############################################################################
# Generate Report: Please run the first #####4 blocs##### at the top of this file
# Define the problem, # import Libraries, # Open the Dataset, # Clean the Data
with open("create_LaTeX.py") as f:
    code_create_LaTeX = compile(f.read(), "create_LaTeX.py", 'exec')
    exec(code_create_LaTeX)
#############################################################################

