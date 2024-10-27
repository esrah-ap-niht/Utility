
##############################################################################################################
#### Import Packages 
##############################################################################################################


from xgboost import XGBClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns 
import io
import zipfile
import json
import h5py
import matplotlib.ticker as ticker
from matplotlib_scalebar.scalebar import ScaleBar
import time
from itertools import cycle, islice
from sklearn.metrics import adjusted_rand_score, det_curve
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import *
from sklearn.mixture import *
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import RocCurveDisplay, roc_curve, DetCurveDisplay

from tkinter import filedialog
from tkinter import *
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
#from utils import *
#from model import UNET
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 
#from torchvision.transforms import v2
from sklearn.utils.class_weight import compute_class_weight

import gc 
import cv2
import random 
import os
from PIL import Image 
import torch
from torch.utils.data import Dataset
#from torchvision import transforms, datasets
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit
import pickle 
from datetime import datetime
import GPUtil
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd 
import torch
#from model import UNET
#from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from cityscapesscripts.helpers.labels import trainId2label as t2l
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
import statistics as stats
from scipy.stats import mode 
import scipy
from skimage.morphology import skeletonize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
#from datasets import CityscapesDataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from boto3 import Session
import math 

##############################################################################################################
#### Configuration 
##############################################################################################################
# Enable garbage collection for memory 
gc.enable()

# Removes tkinter popups after selecting directories 
try: 
    root = Tk()
    root.withdraw()
    root.attributes('-topmost',1)
except:
    pass 

def cohen_d(a,b):
    np.abs(np.mean(a) - np.mean(b)) / math.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0)
    
    

def cohen_d_descriptors(x,y):
    """
    Calculate descriptive statistics using effect size, one vs rest approach for each unique value of the dependent variable. 

    Parameters
    ----------
    x : TYPE: Dataframe
        DESCRIPTION: Dataframe of independent variables 
    y : TYPE: Dataframe or array
        DESCRIPTION: Dependent variable 

    Returns
    -------
    Dataframe of descriptives. Each row is one column in x. 

    """

    
    columns = list(   x.columns   )
    
    x = np.array(x)
    y = np.array(y)        
    
    unique_dependent_variables = list(   np.unique(y)   )
    running_list = [] 
    output = pd.DataFrame() 
    
    for value in unique_dependent_variables:
        running_list = []
        for i in range(x.shape[1]):
            current_subset = x[y==value, i]
            other_subset   = x[y!=value, i]
    
            running_list.append(   cohen_d( current_subset, other_subset )   )
    
        output[   str(value) + " vs Rest"] = running_list
    
    output['Columns'] = columns
    
    return output 
    
    













        
        
        

def cluster_and_score_similarity(X, y, visualize=True):
    """
    This function inputs an array or dataframe, standardizes by column, and then performs clustering 
    with a variety of algorithms. Cluster predictions are compared to ground truth labels 
    and a Rand Index similarity score calculated for each set of model parameters
    

    Parameters
    ----------
    X : TYPE: dataframe or array 
        DESCRIPTION: Independent variables to be clustered by a variety of methods.
        
    y : TYPE: Dataframe, 1D array, or list. 
        DESCRIPTION: Variable that corresponds to discrete dependent variables.
        Can be an encoded integer or strings. 
        Length must be equal to the number of rows in X

    Returns
    -------
    Rand Index scores (list of floats)
    Model parameters (list of strings)

    """
    
    # Convert to arrays for consistency.
    X = np.array(X)
    y = np.array(y)
    
    if len(y) != X.shape[0]:
        raise Exception("Independent and dependent variables must have the same number of samples")
    
    # Define the models and parameter combinations to be explored 
    models = [AgglomerativeClustering(),
              SpectralClustering(),
              DBSCAN(),
              HDBSCAN(),
              OPTICS(),
              AffinityPropagation(),
              Birch(),
              GaussianMixture()
             ]
    
    number_clusters = [3,4,5,6,7,8,9]
    min_samples = [2,3,5,7,9]
    min_cluster_size = [3,5,7,10]
    
    parameters = [
        {"n_clusters": number_clusters, "linkage": ['ward', 'complete', 'average', 'single']},
        {"n_clusters": number_clusters, "eigen_solver":["arpack"], "affinity": ["nearest_neighbors"] },
        {"eps": [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10]},
        {"min_samples": min_samples, "min_cluster_size": min_cluster_size, "allow_single_cluster": [False]},
        {"min_samples": min_samples, "min_cluster_size": min_cluster_size, "xi": [0.05] }, 
        {"damping": [0.5, 0.6, 0.7, 0.8, 0.9] },
        {"n_clusters": number_clusters, "branching_factor": [3,5,7,9,11,15], "threshold": [0.1, 0.5, 0.9]},
        {"n_components": number_clusters, "covariance_type": ["full"]}
        ]
        
    # Normalize data as best practice 
    X = StandardScaler().fit_transform(X)
    
    # Setup empty lists to store the model scores and model names. 
    # Both lists will be returned at the end of the function. 
    model_params = []
    rand_scores = []
    
    # Iterate through models and then parameters in a grid search 
    for i, model in enumerate(tqdm(models)):
        grid = ParameterGrid(parameters[i] )
        for combination in grid: 
            model.set_params(   **combination  )
            model_text = str(model) # Used for plot titles and for recording scores 
            
            # Fit model 
            y_pred = model.fit_predict(X)
        
            # Save model and parameter combinations for output
            model_params.append(model_text)
            # Save scores for output. 
            rand_scores.append(   adjusted_rand_score(y, y_pred)   )
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn-metrics-adjusted-rand-score
            
            # Optional visualizations
            if visualize:
                cm = 100.0*confusion_matrix(y, y_pred, normalize = 'pred')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap = 'plasma')
                fig = plt.gcf()
                fig.set_size_inches(10,10)
                plt.title("Confusion Matrix - " + str(model_text))
                plt.xticks(rotation=90)
                plt.xlabel("Clustering Prediction")
                plt.ylabel("Original Label")
                plt.show()
                
    return rand_scores, model_params


def heirarchial_train_test_split(df, split_by, stratify_by, test_size, dependent_variable):
    """
    It may be desirable to split datasets by some variable in addition to stratifying. 
    For example, one may not want a given medical patient showing up in both test and train
    datasets in addition to stratifying by diagnosis in the training dataset. 
    
    
    Parameters
    ----------
    df 
        TYPE: DataFrame
        DESCRIPTION: Input dataframe with both independent and dependent variables.
    
    split_by 
        TYPE: String
        DESCRIPTION: Column by which to split into testing or training, but not both. 
    
    stratify_by 
        TYPE: String
        DESCRIPTION: Column by which to stratify, conceptually identical to sklean's train_test_split.  
    
    test_size 
        TYPE: Float
        DESCRIPTION: Value between 0.0 and 1.0 that specifies what percentage of the "split_by" unique values are assigned to
                     either the test or train dataset. Note - test and training dataframes may not have number of rows in the 
                     percentage specified by test_size. 
    
    dependent_variable 
        TYPE: String
        DESCRIPTION: Column name which has the dependent variable. 
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    
    Dataframes of testing and training data. 
    
    """
    
    # Create a list of all unique values in the column that is being split.
    unique_values = list(np.unique(   df[split_by]   ))
    
    # Now take a random sampling to get the values in the test dataset.
    number_samples = int(   test_size*len(unique_values)   )
    test_values = random.sample(unique_values,number_samples)
    
    # Now iterate through the list of all possible values and assign
    # any that do not appear in the test group to the training dataset. 
    train_values = [] 
    for value in unique_values:
        if value in test_values:
            pass 
        else:
            train_values.append(value) 
    
    # Split dataframes using the prepared lists. 
    test_df = df[df[split_by].isin(   list(test_values)   )]
    train_df = df[df[split_by].isin(   list(train_values)   )]
    
    # Also apply stratifying to the training dataset. 
    sample_size = min(np.unique(train_df[stratify_by], return_counts=True)[1])
    train_df = train_df.groupby(stratify_by, group_keys=False).apply(lambda x: x.sample(   sample_size   ))
    
    # Reset indexes. 
    test_df.reset_index(inplace=True, drop=True)
    train_df.reset_index(inplace=True, drop=True)
    
    # Split off the dependent variables. 
    y_test = test_df.pop(dependent_variable)
    y_train = train_df.pop(dependent_variable)
    
    return train_df, test_df, y_train, y_test

     
def train_classifier_battery(df, split_by, stratify_by, test_size, dependent_variable, visualize=True, n_simulations = 10): 
    """
    This function inputs a dataframe and conducts Monte Carlo simulations to estimate both the mean and variability of 
    different models and hyperparameter combinations in classification. It can be thought of as repeated stratified K-fold 
    with resampling. 
    
    
    Parameters
    ----------
    df  
        TYPE: Dataframe 
        DESCRIPTION: Dataframe containing both dependent and independent variables. 
                    Columns are different variables, rows are different datapoints. 
                    It is assumed that inputation or filling NAN cells has already been conducted. 
        
    split_by 
        TYPE: String
        DESCRIPTION: Column by which to split into testing or training, but not both. 
    
    stratify_by 
        TYPE: String
        DESCRIPTION: Column by which to stratify, conceptually identical to sklean's train_test_split.  
    
    test_size 
        TYPE: Float
        DESCRIPTION: Value between 0.0 and 1.0 that specifies what percentage of the "split_by" unique values are assigned to
                     either the test or train dataset. Note - test and training dataframes may not have number of rows in the 
                     percentage specified by test_size. 
    
    dependent_variable
        TYPE: String
        DESCRIPTION: Column name which has the dependent variable. 
        
    visualize
        TYPE: Boolian
        DESCRIPTION: Whether or not to show graphs and plots.  
        
    n_simulations
        TYPE: Integer 
        DESCRIPTION: Number of Monte Carlo simulations to run. 
        
    Returns
    -------
    None.
    
    Results are saved as CSV files to the local drive. 
    
    """
    
    # Classifier models to evaluate. 
    models = [
             #XGBClassifier(),
             SVC(),
             KNeighborsClassifier(),
             #GaussianProcessClassifier(),
             DecisionTreeClassifier(),
             RandomForestClassifier(),
             #MLPClassifier(),
             AdaBoostClassifier(),
             GaussianNB(),
             QuadraticDiscriminantAnalysis(),
             LogisticRegression()
             ]
    
    # Hyperparameter combinations to evaluate. 
    parameters = [
                 # {'num_class': [len(  np.unique(df[dependent_variable])  )], 'objective': ['multi:softmax']  },
                  {'kernel':['linear', 'rbf'], 'C':[0.0001,0.001,0.01,0.1,0.5,1, 10, 100]},
                  {'n_neighbors': [3,4,5,6,7], 'weights': ('uniform', 'distance')},
                  #{'kernel': [1.0 * RBF(1.0), 1.0 * RBF(0.1)]},
                  {'max_depth': [2,4,6,8,10], 'min_samples_leaf': [1,3,5], 'splitter': ('best', 'random'), 'class_weight': ['balanced']},
                  {'max_depth': [2,4,6,8,10], 'min_samples_leaf': [1,3,5], 'n_estimators':[10, 50, 100], 'max_features': [1, 3, 5, 10, 'sqrt', 'log2', None]},
                  #{'hidden_layer_sizes': [(10,), (50,), (100,)], 'max_iter': [1000] },
                  {'n_estimators': [10,50,100,500]},
                  {},
                  {},
                  {'C':[0.0001,0.001,0.01,0.1,0.5,1, 10, 100] }
                  ]
    
    # Used later. This way it is only computed once. 
    number_unique_dependent_values = len(np.unique(df[dependent_variable]))
    
    # Encode dependent variable labels of the entire dataset. 
    y = df[dependent_variable]
    le = LabelEncoder()
    le.fit(y)
    
    # Setup variables to store metrics. 
    metrics = pd.DataFrame()
    output =[]
    best_accuracy = 0
    best_clf = None
    
    # Iterate through models, then hyperpameter combinations, then simulation iterations. 
    for i, model in enumerate(tqdm(models)):
        grid = ParameterGrid(parameters[i] )
        model_text = str(model)
        
        for combination in tqdm(grid): 
            model.set_params(   **combination  )
            
            # User feedback to monitor progress. 
            print(str(model))
            
            clf = make_pipeline(StandardScaler(), model)
            
            total_accuracy = [] 
            AUC = [] 
            accuracy_true_negative = []
            accuracy_true_positive = [] 
            for j in range(n_simulations): 
                    
                try:
                    while True:
                        # If the "split_by" variable is used, heirarchically split data. Otherwise use the normal sklearn function. 
                        if split_by is not None:
                            X_train, X_test, y_train, y_test = heirarchial_train_test_split(df, split_by = split_by, stratify_by = stratify_by, test_size = test_size, dependent_variable = dependent_variable)
                            X_train.pop(split_by)
                            X_test.pop(split_by)
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(df, df[dependent_variable], test_size = test_size, shuffle = True, stratify = df[stratify_by])
                        
                        if set(np.unique(y_test)) == set(np.unique(y_train)):
                            break 
                    
                    # This can be uncommented for debugging purposes to verify that there is no overlap if the "split_by" is used. 
                    #set(np.unique( X_test[split_by] )) &  set(np.unique( X_train[split_by] ))
                    
                    # Encode
                    y_test = le.transform(y_test)
                    y_train = le.transform(y_train)
                    
                    # Train model and predict values for test dataset
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    
                    # Score overall accuracy. 
                    try:
                        total_accuracy.append(100.0*balanced_accuracy_score(y_test, y_pred))
                    except:
                        total_accuracy.append(0)
                        
                    # Score AUC if possible. Not all models in Sklearn permit AUC.
                    try:
                        AUC.append(roc_auc_score(y_test, clf.decision_function(X)))
                    except:
                        AUC.append(0) 
                    
                    # Special case of binary predictions. 
                    if number_unique_dependent_values == 2: 
                        cm = 100.0*confusion_matrix(y_test, y_pred, normalize = 'true')
                        try:
                            accuracy_true_negative.append(cm[1,1])
                            accuracy_true_positive.append(cm[0,0])
                        except:
                            accuracy_true_positive.append(0)
                            accuracy_true_negative.append(0) 
                except:
                    pass 
            
            if len(accuracy_true_positive) == 0:
                continue 
            
            # Save simulation results. 
            # Because Pandas is not computationally efficient, it is better to save results
            # at the end of each simulation batch rather than at the end of each simulation. 
            temp_df = pd.DataFrame()
            
            if number_unique_dependent_values == 2: 
                temp_df['Accuracy'] = accuracy_true_positive
                temp_df['Model'] = model_text
                temp_df['Parameter'] = str(combination)
                temp_df['Metric'] = "True Positive"
                metrics = pd.concat([metrics, temp_df], axis = 0)
                
                temp_df['Accuracy'] = accuracy_true_negative
                temp_df['Model'] = model_text
                temp_df['Parameter'] = str(combination)
                temp_df['Metric'] = "True Negative"
                metrics = pd.concat([metrics, temp_df], axis = 0)
            
            temp_df['Accuracy'] = total_accuracy
            temp_df['Model'] = model_text
            temp_df['Parameter'] = str(combination)
            temp_df['Metric'] = "Accuracy"
            metrics = pd.concat([metrics, temp_df], axis = 0)

            temp_df['Accuracy'] = AUC
            temp_df['Model'] = model_text
            temp_df['Parameter'] = str(combination)
            temp_df['Metric'] = "AUC"
            metrics = pd.concat([metrics, temp_df], axis = 0)
            # Append batch results to the "master" dataframe of metrics. 
            
            
            metrics['Model Parameter Combination'] = metrics['Model'] + metrics['Parameter']
            metrics['Model'] = metrics['Model'].astype(str)
            metrics.reset_index(inplace=True, drop=True)
            
            label = metrics['Model Parameter Combination'][-1:]
            label.reset_index(drop= True, inplace=True)
            label = label[0]
        
            # Compute the average accuracy for the batch of simulations. 
            #Accuracy = np.mean(metrics[   (metrics['Model Parameter Combination'] == label) & (metrics['Metric'] == "Accuracy")   ]['Accuracy'])  
            Accuracy = np.mean(total_accuracy)
            AUC_score_mean = np.mean(AUC)
            
            try:
                # In the case of binary classification. 
                #TP= np.mean(metrics[   (metrics['Model Parameter Combination'] == label) & (metrics['Metric'] == "True Positive")   ]['Accuracy']) 
                #TN= np.mean(metrics[   (metrics['Model Parameter Combination'] == label) & (metrics['Metric'] == "True Negative")   ]['Accuracy'])  
                TP= np.mean(accuracy_true_positive)
                TN= np.mean(accuracy_true_negative)
                
                output.append([TP,TN,Accuracy, AUC_score_mean, label])
            except:
                # Multiclass 
                output.append( [Accuracy, AUC_score_mean, label] ) 
            
            # If the model and parameter set yields the highest total accuracy, save that combination for later use. 
            if Accuracy > best_accuracy:
                best_clf = clf
                best_accuracy = Accuracy
                best_label = label 
                
                
    # Now that we have iterated through all models and parameter combinations, we will take the model with
    # the highest overall accuracy and create visuializations for model metrics. 
    if number_unique_dependent_values == 2:  
        
        
        while True:
            # If the "split_by" variable is used, heirarchically split data. Otherwise use the normal sklearn function. 
            if split_by is not None:
                X_train, X_test, y_train, y_test = heirarchial_train_test_split(df, split_by = split_by, stratify_by = stratify_by, test_size = test_size, dependent_variable = dependent_variable)
                X_train.pop(split_by)
                X_test.pop(split_by)
            else:
                X_train, X_test, y_train, y_test = train_test_split(df, df[dependent_variable], test_size = test_size, shuffle = True, stratify = df[stratify_by])
            
            # Verify that we have examples of all classes in both training and testing. 
            if set(np.unique(y_test)) == set(np.unique(y_train)):
                break 
             
        # Encode 
        y_test = le.transform(y_test)
        y_train = le.transform(y_train)
        
        # Train a model again. This ensures the ROC curves are plotted for the correct model and test train split. 
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)
        
        # Calculate the ROC and Detection Error Tradeoff plots 
        fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        
        RocCurveDisplay.from_estimator(best_clf, X_test, y_test, ax=ax_roc, plot_chance_level=True)
        DetCurveDisplay.from_estimator(best_clf, X_test, y_test, ax=ax_det)
        
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html#sklearn.metrics.det_curve
        """
        ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
        ax_det.set_title("Detection Error Tradeoff (DET) curves")
        
        ax_roc.grid(linestyle="--")
        ax_det.grid(linestyle="--")
        
        fig.suptitle( "Positive = " + str(le.inverse_transform([1])[0]) )
        plt.savefig("Best Model ROC.jpg", format='jpg', dpi=300)
        plt.show()
        
        try:
            # Now calculate the true false and true negative rates and plot as a table. 
            scores = clf.predict_proba(X_test)[:,1]   
            fpr, fnr, thresholds = det_curve(y_test, scores)
            
            # Convert from false to true positive and negative. 
            tpr = 1.0-fpr
            tnr = 1.0-fnr
            
            # Change back to string labels for plotting. 
            y_test = le.inverse_transform(y_test)
            
            # Draw a diagonal line for 50-50 random chance reference. 
            plt.plot([[1], [0]])
            
            # Plot tradeoffs. 
            plt.plot(1-fnr, 1-fpr)
            plt.ylabel("True Negative Rate: " + str(np.unique(y_test)[0]))
            plt.xlabel("True Positive Rate: " + str(np.unique(y_test)[1]))
            
            # Now take the same data and present as a table, at specified thresholds. 
            tradeoff = np.zeros(shape = (8,2))
            for i, target_value in enumerate([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
                index = np.argmin(np.abs(np.array(tpr)-target_value))
                tradeoff[i, 1] = tpr[index]
                tradeoff[i, 0] = tnr[index]
        
            tradeoff = np.around(tradeoff, decimals = 2)
            labels = [   np.unique(y_test)[0], np.unique(y_test)[1]  ]
            table = plt.table(
                            cellText=tradeoff,
                            colLabels=labels,
                            loc = 'right'
                            )
            plt.show() 
        except:
            pass 
        

    else:
        # In the case of multi-label classifiers 
        fig, ax_roc = plt.subplots(1, 1, figsize=(8.5, 5), constrained_layout=True)
        RocCurveDisplay.from_estimator(best_clf, X_test, y_test, ax=ax_roc, plot_chance_level=True)
        ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
        ax_roc.grid(linestyle="--")
        fig.suptitle(best_label)
        plt.savefig("Best Model ROC.jpg", format='jpg', dpi=300)
        plt.show()
    
    # Optional KDE plots to visualize model metrics 
    if visualize:
        # All models and parameters together
        pal = sns.color_palette(palette='tab10')
        g = sns.FacetGrid(metrics, 
                          col="Metric", 
                          hue="Model Parameter Combination", 
                          aspect=1.5, 
                          height=3, 
                          palette=pal, 
                          sharey=False)
        
        g.map(sns.kdeplot, "Accuracy",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=0.75, linewidth=.5,
              warn_singular=False)
        g.set_xlabels('Relative Frequency')
        g.set_xlabels('Accuracy (%)')
        g.fig.subplots_adjust(top=0.8) 
        g.fig.suptitle('Density Plots of Simulated Predictions')
        plt.show() 
        
        # Split different models into rows
        pal = sns.color_palette(palette='tab10')
        g = sns.FacetGrid(metrics, 
                          col = "Metric", 
                          row = "Model",
                          hue = "Model Parameter Combination", 
                          aspect = 1.5, 
                          height = 3, 
                          palette = pal, 
                          sharey = False,
                          sharex=False)
        
        g.map(sns.kdeplot, "Accuracy",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=0.75, linewidth=.5,
              warn_singular=False)
        g.set_xlabels('Relative Frequency')
        g.set_xlabels('Accuracy (%)')
        g.fig.subplots_adjust(top=0.8) 
        g.fig.suptitle('Density Plots of Simulated Predictions')
        plt.show() 
        
    # We have computed the average metrics across batches of simulations and saved said averages as a list. 
    # Now convert to a dataframe. 
    if number_unique_dependent_values == 2: 
        output = pd.DataFrame(output, columns=['Mean True Positive', 'Mean True Negative', 'Accuracy', 'AUC', 'Model'])
    else:
        output = pd.DataFrame(output, columns=['Accuracy', 'Model'])

    # Sort and save to local drive. 
    output = output.sort_values(by=['Accuracy'], ascending=False)
    output.reset_index(inplace=True) 
    metrics.reset_index(inplace=True) 
    
    output.to_csv('Mean model simulation results.csv', index = False)
    metrics.to_csv('Monte Carlo simulation results.csv', index = False)
    
    return best_clf

     






