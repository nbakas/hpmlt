


import os
import zipfile
import getpass
import shutil
import datetime



def write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_):
    if os.path.exists(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + file_name + __extension__):
        image_paths.append(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + file_name + __extension__)
        latex_code += __text__
        latex_code += '\n\n\n'
        latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=' + str(scale_) + '\\textwidth]{' 
        latex_code +=  __section__ + "/" + sub_dir + "/" + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
        latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
        latex_code += '\n\n\n'
    return image_paths, latex_code

def write_figure_sub_sub_dir(ROOT_DIR, __section__, sub_dir, sub_sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_):
    if os.path.exists(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + sub_sub_dir + os.sep + file_name + __extension__):
        image_paths.append(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + sub_sub_dir + os.sep + file_name + __extension__)
        latex_code += __text__
        latex_code += '\n\n\n'
        latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=' + str(scale_) + '\\textwidth]{' 
        latex_code +=  __section__ + "/" + sub_dir + "/" + sub_sub_dir + "/" + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
        latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
        latex_code += '\n\n\n'
    return image_paths, latex_code

####################################################################################################################
# Set the name of the LaTeX output file and the zip file
project_name = os.path.basename(os.path.normpath(ROOT_DIR)) 
tex_file = project_name + '.tex'
zip_file = project_name + '.zip'
__extension__ = ".png"
shutil.copyfile("refs.bib", ROOT_DIR + "LaTeX_report" + os.sep + "refs.bib")
date_string = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
image_paths = []
features_names_latex = []
for f in features_names:
    features_names_latex.append(f.replace("_","\\_").replace("#","\\#"))
target_name_latex = target_name.replace("_","\\_").replace("#","\\#")
####################################################################################################################



####################################################################################################################
# Initialize the LaTeX code and a list to store the image file paths
latex_code = '\\documentclass{article}\n'
latex_code += '\\usepackage{graphicx}\n'
latex_code += '\\usepackage[a4paper, total={6.5in, 9.0in}]{geometry}\n'
latex_code += '\\usepackage{hyperref}\n'
latex_code += '\\hypersetup{\n'
latex_code += 'colorlinks=true,\n'
latex_code += 'linkcolor=blue,\n'
latex_code += 'urlcolor=blue,\n'
latex_code += 'citecolor=blue,\n'
latex_code += '}\n'
latex_code += '\\urlstyle{same}\n'
latex_code += '\\usepackage{url}\n'
latex_code += '\\usepackage{float}\n'
latex_code += '\\usepackage{longtable}\n'
latex_code += '\\usepackage{dirtytalk}\n'
latex_code += '\n\n\n'
latex_code += '\\title{' + project_name + '}\n'
latex_code += '\\author{' + getpass.getuser() + '}\n'
latex_code += '\\date{' + date_string + '}\n'
latex_code += '\n'
latex_code += '\\begin{document}\n'
latex_code += '\\maketitle\n'
latex_code += '\\tableofcontents\n'
latex_code += '\n'
latex_code += '\n'
latex_code += '\n'
latex_code += '\n'
####################################################################################################################




####################################################################################################################
latex_code += '\\section{Introduction}\n'
latex_code += 'This is a report for Analysis of the ' + project_name + ' dataset, with Machine learning Algorithms. '
latex_code += 'The dataset comprises ' + str(len(features_names_latex)) + ' variables, which are: '
for fe in features_names_latex:
    latex_code += fe +', '
latex_code = latex_code[:-2]
latex_code += '. The target variable is ' + target_name_latex + '.'
latex_code += '\n\n\n\n\n'
latex_code += 'For the preparation of the numerical results, we use the well known hpmlt package \\cite{hpmlt}, '
latex_code += 'as well as the corresponding machine learning algorithms \\cite{'
latex_code += 'annbn,Bakas2019,ALHAMAYDEH2022114441,markou2021fundperiod,markouTechnoPress,emcis2,lavdasmetrop,Faggion2017,FaggionJr2018'
latex_code += '}, as described in the following sections.'
latex_code += '\n\n\n\n\n'
####################################################################################################################






####################################################################################################################
__section__ = "Descriptive_Statistics"
latex_code += '\\section{Descriptive Statistics}\n'
latex_code += '\n\n\n'
####################################################################################################################


##########################################################
sub_dir = 'PDF_CDF' + os.sep 
file_name = target_name_latex + '_Train_PDF_CDF'
__extension__ = ".png"
if os.path.exists(ROOT_DIR + __section__ + os.sep + sub_dir + file_name + __extension__):
    image_paths.append(ROOT_DIR + __section__ + os.sep + sub_dir + file_name + __extension__)
    latex_code += '\\subsection{Statistical Properties of Target Variable}\n'
    latex_code += 'In Figure \\ref{fig:' + file_name + '}, the histogram of the target variable is presented.'
    latex_code += '\n\n\n'
    latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{' 
    latex_code +=  __section__ + "/" + sub_dir.replace('\\','/') + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
    latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
    latex_code += '\n\n\n'
########################################################## 


##########################################################
if os.path.exists(ROOT_DIR + __section__ + os.sep + 'descriptive_statistics_train.txt'):
    latex_code += '\\subsection{Statistical Properties of Features}\n'
    with open(ROOT_DIR + __section__ + os.sep + 'descriptive_statistics_train.txt', 'r') as file:
        latex_code += file.read()
        latex_code += '\n\n\n'
##########################################################  


##########################################################
file_name = "All_by_All_Correlation_Cuthill_McKee"
__extension__ = ".png"
if os.path.exists(ROOT_DIR + __section__ + os.sep + file_name + __extension__):
    latex_code += '\\subsection{Correlations}\n'
    
    image_paths.append(ROOT_DIR + __section__ + os.sep + file_name + __extension__)
    latex_code += 'In Figure \\ref{fig:' + file_name + '}, the 90\\% highest correlations are presented. '
    latex_code += 'The correlation matrix has been rearranged with the Cuthill McKee algorithm \\cite{cuthill1969reducing}, '
    latex_code += 'after deleting Pearson correlations less than 90\\%. '
    latex_code += 'This reduces the bandwidth of the matrix and tends to gather groups of correlated variables \\cite{koutsantonis2022bibliometric}.'
    latex_code += '\n\n\n'
    latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{' 
    latex_code +=  __section__ + "/" + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
    latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
    latex_code += '\n\n\n'
    
    latex_code += 'Furthermore, In Table \\ref{tab:All_by_All_Correlation_Cuthill_McKee}, the 20 highest correlations are presented. '
    latex_code += 'The correlations regard all pairs of features and the target variable. In \\textbf{bold} is the target variable. '
    latex_code += 'It also offers a clear image of the highest correlations.'
    latex_code += '\n\n\n'
    with open(ROOT_DIR + __section__ + os.sep + 'All_by_All_Correlation_Cuthill_McKee.txt', 'r') as file:
        latex_code += file.read()
##########################################################


##########################################################
file_name = "map"
__extension__ = ".png"
if os.path.exists(ROOT_DIR + __section__ + os.sep + file_name + __extension__):
    latex_code += '\\subsection{Correlations\' Map}\n'
    image_paths.append(ROOT_DIR + __section__ + os.sep + file_name + __extension__)
    latex_code += 'In Figure \\ref{fig:' + file_name + '}, the correlations map is presented '
    latex_code += 'via a multidimensional scaling technique, presented in '
    latex_code += '\\cite{bakas2019optimal,koutsantonis2022bibliometric,app11115053,iisa2022,Papadaki1}.'
    latex_code += '\n\n\n'
    latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{' 
    latex_code +=  __section__ + "/" + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
    latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
    latex_code += '\n\n\n'
##########################################################  





##########################################################
__section__ = "ML_Models"
latex_code += '\\section{Machine Learning Models}\n'
latex_code += 'We use Least Square Regression as a baseline model, and compare with five more machine learning models'
latex_code += ', all with hyperparameter tuning. Particularly, we train the following models:'
latex_code += 'Polynomial Regression \\cite{Bakas2019,bakas2019taylor,markouTechnoPress,ALHAMAYDEH2022114441,markou2021fundperiod,weisstein2002Poly}, '
latex_code += 'XGBoost \\cite{Chen:2016:XST:2939672.2939785,markou2021fundperiod}, '
latex_code += 'Random Forests \\cite{breiman2001random,ALHAMAYDEH2022114441,markou2021fundperiod}, as implemented in \\cite{scikit-learn,sklearn_api}, '
latex_code += 'Deep Learning with the PyTorch framework \\cite{paszke2019pytorch}, and '
latex_code += 'Artificial Neural Networks with local approximation in partitions (ANNBN) \\cite{annbn}. '
latex_code += 'For the numerical experiments we use the hpmlt \\cite{annbn} software.'
latex_code += '\n\n\n'
##########################################################

titles_ML = ['Linear Regression', 'Polynomial Regression', 'XGBoost', 'Random Forests', 'Deep Learning', 'ANNBN']
##########################################################
for iter, sub_dir in enumerate(['LinRegr', 'NLRegr', 'XGBoost', 'RF', 'DANN', 'ANNBN']):
    latex_code += '\\subsection{' + titles_ML[iter] + '}\n'
    ##########################################################
    

    ##########################################################
    file_name = sub_dir + '_Target_vs_Predicted_Test'
    __text__ = 'In Figure \\ref{fig:' + file_name + '}, the Scatter plot of the Target variable vs Predicted with ' + sub_dir + ' is presented.'
    scale_ = 0.55
    image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
    ########################################################## 


##########################################################
latex_code += '\\subsection{Error Analysis}\n'
latex_code += 'In Table \\ref{tab:All_ML_metrics}, the Accuracy Metrics for all Machie Learning Models '
latex_code += 'are presented in the Train and Test Sets.'
latex_code += '\n\n\n'
with open(ROOT_DIR + __section__ + os.sep + 'All_ML_metrics.txt', 'r') as file:
    latex_code += file.read()
latex_code += '\n\n\n'
##########################################################
with open(ROOT_DIR + __section__ + os.sep + "all_models_sorted_by_mae.txt") as f:
    all_models_sorted_by_mae = f.readlines()[1:]
best_model =all_models_sorted_by_mae[0][:-1]
sub_dir = best_model
sub_sub_dir = 'Error_Analysis'
scale_=1.0
file_name = sub_dir + '_Errors_Test'
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the scatter plot of ' + target_name_latex + ' vs corresponding errors in Test Set for the ' + sub_dir + ' model is presented.'
image_paths, latex_code = write_figure_sub_sub_dir(ROOT_DIR, __section__, sub_dir, sub_sub_dir, 
                                                   file_name, __extension__, __text__, image_paths, latex_code, scale_)
latex_code += '\n\n\n'
##########################################################
file_name = sub_dir + '_Errors_Test_PDF_CDF'
__text__ = "The best model found was " + best_model + ". "
__text__ += 'In Figure \\ref{fig:' + file_name + '}, the histogram of the errors of the ' + sub_dir + ' model in the Test Set is presented.'
image_paths, latex_code = write_figure_sub_sub_dir(ROOT_DIR, __section__, sub_dir, sub_sub_dir, 
                                                   file_name, __extension__, __text__, image_paths, latex_code, scale_)
latex_code += '\n\n\n'
########################################################## 



##########################################################
latex_code += '\\section{Features Importance}\n'
latex_code += '\n\n\n'
##########################################################


##########################################################
latex_code += '\\subsection{p-Values}\n'
latex_code += '\n\n\n'
__section__ = "ML_Models"
sub_dir = "LinRegr"
file_name = "p_Values"
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the sorted p Values of all variables are presented.'
scale_ = 0.95
image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
##########################################################


##########################################################
latex_code += '\\subsection{Features Importance XGBoost}\n'
latex_code += '\n\n\n'
__section__ = "ML_Models"
sub_dir = "XGBoost"
file_name = "features_importance_xgb"
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the features importance with the XGBoost model is presented.'
scale_ = 0.75
image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
##########################################################


##########################################################
latex_code += '\\subsection{Features Importance Random Forests}\n'
latex_code += '\n\n\n'
__section__ = "ML_Models"
sub_dir = "RF"
file_name = "features_importance_rf"
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the features importance with the Random Forests model is presented.'
scale_ = 0.75
image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
##########################################################




##########################################################
latex_code += '\\subsection{Sensitivity Analysis}\n'
latex_code += '\n\n\n'
__section__ = "ML_Models"
sub_dir = "ALL_Sensitivity"
file_name = "All_Sensitivity_XGBoost"
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the maximum minus minimum fluctuation of ' + target_name_latex 
__text__ += ', with stochastic perturbation of features \\cite{emcis2,lavdasmetrop}, with the XGBoost model is presented.'
scale_ = 0.75
image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
##########################################################




##########################################################
# with open(ROOT_DIR + "ML_Models" + os.sep + "ALL_Sensitivity" + os.sep + "important_features.txt", "r") as f:
#     important_features = f.readlines()
# latex_code += '\n\n\n'
# __section__ = "ML_Models"
# sub_dir = "ALL_Sensitivity"
# for i in range(len(important_features)):
#     file_name = "sensitivity_curve_" + important_features[i][:-1]
#     __text__ = 'In Figure \\ref{fig:' + file_name + '}, the Sensitivity Curves for \say{'+ important_features[i][:-1].replace("_","\\_").replace("#","\\#")
#     __text__ += '} with all Machine Learning models are presented.'
#     scale_ = 0.95
#     image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
##########################################################



##########################################################
latex_code += '\\section{Interpretable AI}\n'
latex_code += '\n\n\n'
latex_code += '\\subsection{Decision Tree \& Changes in PDF}\n'
__section__ = "Descriptive_Statistics"
sub_dir = 'Tree' + os.sep 
file_name = target_name_latex + '_tree_with_pdf_for_all_leaves_depth_3'
__extension__ = ".png"
if os.path.exists(ROOT_DIR + __section__ + os.sep + sub_dir + file_name + __extension__):
    image_paths.append(ROOT_DIR + __section__ + os.sep + sub_dir + file_name + __extension__)
    latex_code += 'In Figure \\ref{fig:' + file_name + '}, the Decision Tree the for ' + target_name_latex + ' is presented, in 3 levels, '
    latex_code += 'where we may also see how the Propabiblity Density Functions of ' + target_name_latex 
    latex_code += ' changes for each leaf node, according to the decision rules.'
    latex_code += '\n\n\n'
    latex_code += '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=1.0\\textwidth]{' 
    latex_code +=  __section__ + "/" + sub_dir.replace('\\','/') + file_name + __extension__ + '}\n\\caption{' + file_name.replace("_"," ") 
    latex_code +=  '}\n\\label{fig:' + file_name + '}\n\\end{figure}\n'
    latex_code += '\n\n\n'
########################################################## 


##########################################################
latex_code += '\\subsection{Predictive Equation with Polynomial Regression}\n'
latex_code += '\n\n\n'
__section__ = 'ML_MOdels'
sub_dir = 'NLRegr'
if os.path.exists(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + "__formula__.txt"):
    latex_code += "The Predictive Equation derived Polynomial Regression is in the form of:\n\\par\n"
    latex_code += "$"
    with open(ROOT_DIR + __section__ + os.sep + sub_dir + os.sep + "__formula__.txt") as f:
            equation_ = f.read().replace("_","\\_").replace("#","\\#")
    for qq in range(100):
        equation_ = equation_.replace("*1*","*")
        equation_ = equation_.replace("*1+","+")
        equation_ = equation_.replace("*1-","-")
    equation_ = equation_.replace("*"," \\times ")
    latex_code += equation_
    latex_code += "$"
    latex_code += '\n\n\n'
##########################################################



##########################################################
latex_code += '\\section{Data Centric AI}\n'
latex_code += '\n\n\n'
latex_code += '\\subsection{Errors in Test Set vs Density of Samples}\n'
latex_code += '\n\n\n'
__section__ = "ML_Models"
sub_dir = best_model
file_name = 'MAE_per_bin_' + sub_dir
__text__ = 'In Figure \\ref{fig:' + file_name + '}, the Mean Absolute Error (MAE) in the Test Set is presented, in relation to '
__text__ += 'the cooresponding number of samples of the Train Set found in each bin.'
scale_ = 1.0
image_paths, latex_code = write_figure_sub_dir(ROOT_DIR, __section__, sub_dir, file_name, __extension__, __text__, image_paths, latex_code, scale_)
latex_code += '\n\n\n'
##########################################################




##########################################################
latex_code += '\\section{Predict}\n'
latex_code += '\n\n\n'
latex_code += '\\subsection{Best Model - }\n'
latex_code += '\n\n\n'
latex_code += '\\subsection{Quantile ML Regression - }\n'
latex_code += '\n\n\n'
##########################################################




####################################################################################################################
latex_code += '\n\n\n\n\n'
latex_code += '\\clearpage\n'
latex_code += '\\phantomsection\n'
latex_code += '\\addcontentsline{toc}{section}{References}\n'
latex_code += '\\bibliographystyle{IEEEtran}\n'
latex_code += '\\bibliography{refs}\n\n\n'
latex_code += '\\end{document}'
####################################################################################################################


####################################################################################################################
# Write the LaTeX code to the output file
with open(ROOT_DIR + "LaTeX_report" + os.sep + tex_file, 'w') as f:
    f.write(latex_code)

# Create a zip file containing all of the image files and the LaTeX file
with zipfile.ZipFile(ROOT_DIR + "LaTeX_report" + os.sep + zip_file, 'w') as zip:
    # Add the LaTeX file to the zip file
    zip.write(ROOT_DIR + "LaTeX_report" + os.sep + tex_file, tex_file)
    zip.write(ROOT_DIR + "LaTeX_report" + os.sep + "refs.bib", "refs.bib")
    
    # Add each image file to the zip file
    for image_path in image_paths:
        # Remove the main directory from the image path to create a relative path
        rel_path = os.path.relpath(image_path, ROOT_DIR)
        zip.write(image_path, rel_path)

print('Report has successfully been generated!')



# import subprocess
# cwd = os.getcwd()
# os.chdir(ROOT_DIR + "LaTeX_report" + os.sep + project_name + os.sep)
# subprocess.run(['pdflatex', tex_file])
# # If your document has a bibliography, run bibtex
# subprocess.run(['bibtex', 'main'])
# # Run pdflatex again to update the references
# subprocess.run(['pdflatex', 'main.tex'])
# # Finally, open the PDF file
# subprocess.run(['open', 'main.pdf'])
# os.chdir(cwd)






