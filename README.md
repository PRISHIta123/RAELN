# RAELN (L2-Regularized Autoencoder Enabled Ladder Networks Classifier)

This repository contains the code implementation for the paper: **[A New Combined Model with Reduced Label Dependency for Malware Classification](https://www.atlantis-press.com/proceedings/iciic-21/125960833)** (Prishita Ray, Tanmayi Nandan, Lahari Anne, Kakelli Anil Kumar), published in Proceedings of the 3rd International Conference on Integrated Intelligent Computing Communication & Security (ICIIC 2021)

**Abstract:** With the technological advancements in recent times, security threats caused by malware are increasing with no bounds. The first step performed by security analysts for the detection and mitigation of malware is its classification. This paper aims to classify network intrusion malware using new-age machine learning techniques with reduced label dependency and identifies the most effective combination of feature selection and classification technique for this purpose. The proposed model, L2 Regularized Autoencoder Enabled Ladder Networks Classifier (RAELN-Classifier), is developed based on a combinatory analysis of various feature selection techniques like FSFC, variants of autoencoders and semi-supervised classification techniques such as ladder networks. The model is trained and tested over UNSW-NB15 and benchmark NSL-KDD datasets for accurate real time model performance evaluation using overall accuracy as well as per-class accuracy and was found to result in higher accuracy compared to similar baseline and state-of-the-art models.  

All code files are available under the ./src  
Associated plots are present under ./Plots  
The datasets including the benchmark NSL KDD dataset are present in ./data

UNSW-NB15 Dataset Link: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/  
NSL-KDD Dataset Link: https://www.kaggle.com/hassan06/nslkdd#__sid=js0  

Note: This project is currently compatible with tensorflow v1.15. The code will be upgraded to support the latest version of Tensorflow/Pytorch.    

## Installation  

In order to support tensorflow v1.15, your local system should have a Python version <=3.7 installed, with the installation directory added to your PATH.  

Clone the project repository by downloading the zip file or by using:  
```git clone https://github.com/PRISHIta123/RAELN.git```  

Open a command prompt/terminal and navigate to the folder containing the cloned repository.  

Create a virtual environment to run the project using virtualenv (see this [link](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) for installing virtualenv):  
```virtualenv -p /path/to/Python3.7/python.exe env ```  

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

## Citation  

If you use this paper/code in your research, please consider citing us:

```
@inproceedings{Ray2021,
  title={A New Combined Model with Reduced Label Dependency for Malware Classification},
  author={Prishita Ray and Tanmayi Nandan and Lahari Anne and Kakelli Anil Kumar},
  year={2021},
  booktitle={Proceedings of the 3rd International Conference on Integrated Intelligent Computing Communication & Security (ICIIC 2021)},
  pages={23-32},
  issn={2589-4919},
  isbn={978-94-6239-428-5},
  url={https://doi.org/10.2991/ahis.k.210913.004},
  doi={https://doi.org/10.2991/ahis.k.210913.004},
  publisher={Atlantis Press}
}
```








