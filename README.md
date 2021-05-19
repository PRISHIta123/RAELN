# Combined_Model_with_Reduced_Label_Dependency_for_Classification_of_Network_Intrusion_Malwares

This repository contains the code implementation for the paper: A New Combined Model with Reduced Label Dependency for Malware Classification (Prishita Ray, Tanmayi Nandan, Lahari Anne, Kakelli Anil Kumar), accepted at the ICIIC Conference 2021.  

All code files are available under the ./src  
Associated plots are present under ./Plots  
The datasets including the benchmark NSL KDD dataset are present in ./data

UNSW-NB15 Dataset Link: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/  
NSL-KDD Dataset Link: https://www.kaggle.com/hassan06/nslkdd#__sid=js0  

Note: This project is currently compatible with tensorflow v1.15. The code will be upgraded to support the latest version of Tensorflow/Pytorch.    

## Installation  

In order to support tensorflow v1.15, your local system should have a Python version <=3.5 installed, with the installation directory added to your PATH.  

Clone the project repository by downloading the zip file or by using:  
```git clone https://github.com/PRISHIta123/RAELN.git```  

Open a command prompt/terminal and navigate to the folder containing the cloned repository.  

Create a virtual environment to run the project using virtualenv (see this [link](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) for installing virtualenv):  
```virtualenv -p /path/to/Python3.5/python.exe env ```  

Use these commands to activate the virtual environment.  
For Windows Users: ```\env\Scripts\Activate```  
For Mac OS/Linux Users: ```source /env/Scripts/Activate ```  

Install the project dependencies using:  
```pip install -r requirements.txt```  

## Usage  

To perform Principle Component Analysis on a dataset to report optimal number of features for representation, navigate into the src subdirectory and run the following with the dataset argument:  
```python PCA_Components.py --dataset=DATASET```  

To run a model, navigate into the src subdirectory and pass the dataset, feature selector and classifier arguments as follows:   
```python main.py --dataset=DATASET --feature_selector=FEATURE_SELECTION_ALGO --classifier=CLASSIFICATION_ALGO```  

possible values of the arguments:  
```DATASET```: "UNSWNB15", "NSLKDD"  
```FEATURE_SELECTION_ALGO```: "AE","AEL1","AEL2","AEDropout","FSFC", "None"  
```CLASSIFICATION_ALGO```: "LadderNW", "LadderNWsup", "RandomForest", "NaiveBayes"  

The number of components vs dataset variance for PCA, training loss vs epochs for feature selection, feature importances, chosen features, training and testing (overall and per class) accuracies will be reported. Predicted vs Actual classes on the testing set can be accessed at the /output/predictions.csv file after running the model.  






