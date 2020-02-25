# Semi_Supervised_Model_for_Classification_of_Network_Intrusion_Malwares

All source code files are available in the src folder  
Associated plots are present in the Plots folder  
The datasets including the benchmark NSL KDD dataset is present in the data folder  

UNSW Dataset Link: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/  
NSL-KDD19 Dataset Link: https://www.kaggle.com/hassan06/nslkdd#__sid=js0

Overview of steps involved:
1. Datasets of Network Intrusion Malwares are preprocessed and values are normalized to improve model convergence
2. Unsupervised Feature selection is performed using 4 variants of autoencoders and the FSFC algorithm to reduce overfitting (code can be accessed under src/autoencoder.py, src/autoencoder_l1.py, src/autoencoder_l2.py, src/autoencoder_dropout.py, src/fsfc.py)

Link to research paper describing the FSFC algorithm:
https://www.researchgate.net/publication/328108410_A_new_unsupervised_feature_selection_algorithm_using_similarity-based_feature_clustering

3. Data of selected features is passed through proposed semi-supervised ladder networks model, fully-supervised ladder networks model, Random Forest and Naive Bayes Algorithms for multi-class classification (code can be accessed under src/ladder_nw.py, src/ladder_nw_supervised.py, src/random_forest.py, src/naive_bayes.py)

Link to research paper describing Ladder Networks:
https://arxiv.org/pdf/1507.02672v2.pdf

4. Training and Testing accuracy is recorded for all combinations of the above feature selection algorithms and classification models to report the best combination on each dataset separately.




