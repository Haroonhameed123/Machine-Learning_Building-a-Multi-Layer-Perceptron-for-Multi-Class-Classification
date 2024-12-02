# Machine-Learning_Building-a-Multi-Layer-Perceptron-for-Multi-Class-Classification
Building a Multi-Layer Perceptron for Multi-Class Classification: A Comprehensive Tutorial on Paper, Rock, Scissors Dataset

1. Introduction
In this tutorial, we will use a Multi-Layer Perceptron (MLP) to solve a multi-class classification problem. The dataset has 1000 records with 17 numerical features as input variables and a categorical target column (paper, rock, scissors) as class labels.
The goal of the MLP model is to classify each record into one of the three classes on the basis of the features presented. MLPs are an artificial neural network, which are composed of layers of interconnected nodes. Each node learns the patterns and the relationships from the data. Models are quite powerful for data sets having numerical input features, since they can be used efficiently to handle the non-linear relationships.
This tutorial will cover
•	Data preparation: Feature scaling, label encoding.
•	Designing and training an MLP model. Optimizing the depth of hidden layers to give the best performance.
•	Training and evaluation, evaluating and visualization of results with discussions on training metrics, confusion matrix, and predictions.

2. Objective
This tutorial demonstrates the use of a Multi-Layer Perceptron (MLP) to solve a multi-class classification problem. The goals are broken down into four key steps to ensure a complete understanding of the process:
Preprocess the data
Preprocessing is essential to ensure that the data is in a suitable format for training the neural network. This has two major steps:
Encoding Categorical Labels: This target column is categorical in nature, as it contains the class labels (paper, rock, scissors). To make it more usable for the MLP model, it is converted into numerical labels using Label Encoding. Moreover, the labels are transformed into a one-hot encoded format, which is required for multi-class classification tasks.
Standardizing Numerical Features: StandardScaler is used to standardize the 17 numerical features. This scales every feature to have a mean of 0 and a standard deviation of 1. In this way, all features are equally contributing to the model's learning process, which prevents biasing from the features with larger magnitudes.
Design and Train an MLP Model
The model is designed with two hidden layers and a softmax output layer to accommodate the multi-class nature of the problem. The layers are structured to yield meaningful patterns of the input features.
This includes training the parameters of the model with a dynamic adjustment in the learning rate using Adam, such that convergence is better; it was trained on 80% of the data and made sure 20% was reserved to validate that this model was generalizing appropriately to data the model had never seen before.
Evaluate performance
The trained model is tested on another independent test set that includes 20% of the available data. Accuracy is the amount by which the model generally classifies the instances; a confusion matrix, though, offers more details in the form of true positives, false positives, false negatives, and true negatives for every class. Classification Report provides an accurate summary of key metrics like precision, recall, and F1-score for every class.
These metrics help understand the strengths and weaknesses of the model at different classes.
Visualizations
The two key aspects visualized are
Training Process: Loss and Accuracy Curves for both training and validation sets are plotted to monitor the learning curve and find issues such as overfitting or underfitting.
Model performance: A confusion matrix heatmap will show the distribution of the predictions for each class. Therefore, in detail, it helps understand classification performance.

3. Dataset Overview
The dataset for this tutorial comprises 1000 records and 18 columns as follows:
Features: The first 17 columns feature_1 through feature_17 consist of numerical data that are input features for the model. The feature represents different attributes that would be used by the model to classify data points into one of three categories.
Target: The 18th column (target) is categorical with three classes possible in this, that are paper, rock, and scissors. It is what the model aims to predict.
Data Set Features:
Missing Values: Zero
It means that all the rows or columns of this dataset have no null or missing values. Therefore, preprocessing as well as training is conducted very smoothly and does not require imputation. Target Distribution:
The target column is balanced in its representation of all three classes: paper, rock, and scissors. This makes the dataset fair to train because it is not biased to a particular class.
Dataset Statistics
Mean, Standard Deviation, Min, and Max Values
Every feature is distinct in terms of its statistical properties, which can be observed from the descriptive analysis step. For example,
The average values (mean) of the features are close to zero.
The features are spread across various ranges with different minimum and maximum values.
Standard Deviation:
It is a measure of the spread of feature values that justifies the need for standardization, which brings all features to the same scale.
 
4. Data Preprocessing
Data preprocessing is one of the essential steps involved in machine learning pipelines that guarantees the data is fit to train models on. Steps undertaken include the following:
Separation of Features and Targets:
The dataset was separated into features (X) and the target variable (y).
Features(X): The dataset had 17 numerical columns: feature_1 to feature_17.
Target (y): The categorical column target
This helps differentiate between input variables and the target to be predicted.
Label Encoding:
The categorical target labels are converted to numerical values as follows, using LabelEncoder:
paper = 0
rock = 1
scissors = 2
This step transforms the labels into a format in which the model can process them easily.
Feature Scaling:
Features are standardized by using StandardScaler :
Every feature was rescaled with the mean of 0 and the standard deviation of 1.
Why Scaling?
Since the features differ in range, that is, some might be ranging from -10 to 10 and others between -100 and 100, scaling helps all the features have a chance of equal participation during training.
Train-Test Split:
Divided the data set as
80% Training Data: this data was used to train the model
20% Testing Data: used for evaluation, the performance of the model on unseen data; This makes sure that the model is capable of generalizing and avoiding overfitting.
 


5. Model Design
A good model design is crucial for high performance. In this project, an MLP model was designed for multi-class classification. The following is the detailed description of the model architecture and its compilation:
MLP Architecture
Input Layer
Input 17 features (one for each numerical column).
Hidden Layers
Layer 1
128 neurons with ReLU activation function.
Includes 30% dropout to avoid overfitting by randomly ignoring 30% of the connections during training.
Layer 2
64 neurons with ReLU activation function.
Includes 30% dropout, as in Layer 1, to introduce more regularization.
Output Layer
3 neurons, each representing one class (paper, rock, scissors).
Softmax activation function to output probabilities for each class. This way, the sum of probabilities is 1.
Optimizer
Adam: adaptive learning rate optimizer that makes the training converge faster.
Loss Function
Categorical Crossentropy: For multi-class classification, suitable. Calculates difference between probabilities and true class values.
Evaluation Metric
Accuracy: calculates how well the training model does, as well as its ability to make good predictions at the time of evaluation.

 
 

6. Training the Model
The training process is an essential part of developing an efficient machine learning model. The MLP model was trained with the preprocessed data for 50 epochs with the following parameters and results.
Batch Size
32: This is the number of samples processed before the model updates its weights.
Validation Split
20% of the training data was set aside for validation to keep track of the model's performance while training.
Epochs
The model was trained for 50 epochs to ensure learning without overfitting.
Training Accuracy
•	It increased steadily with each epoch and reached around 86.5% at the end of training.
•	It indicates that the model learned the pattern in the training data properly.
Validation Loss
•	The loss decreased steadily with every epoch.
•	This shows the model generalizes well to unseen data without much overfitting.

7. Evaluation
Evaluation is concerned with checking the performance of the model on the test dataset; this gives insights into its prediction ability and possible areas of improvement.
Results
Test Accuracy
The overall test accuracy for the model was 86.5%, which indicated strong performance across all classes.
 
Confusion Matrix
Paper: High recall of 96% indicates that most paper instances were correctly classified.
Rock: Fair recall (80%) with some misclassifications for scissors and paper.
Scissors: Poor performance, mainly because of its low representation in the dataset, which results in lower precision and recall.
 
Classification Report
It gives the metrics for each class
Paper
Precision: 86%
Recall: 96%
F1-Score: High, which indicates that it has learned well and is balanced.
Rock
Precision: 90%
Recall: 80%
F1-Score: Good predictive performance but slightly low recall.
Scissors
Precision: 79%
Recall: 68%
F1-Score: Lower scores suggest that it needs more training data or a better representation for this class.
 
 

Insights
•	The model is well-performing for all classes with balanced precision and recall for paper and rock.
•	Scissors classification still proves challenging, probably due to class imbalance or overlapping feature distributions.
Precision and Recall
•	High precision ensures that most of the predicted instances for a class are correct.
•	High recall ensures that most of the actual instances of a class are captured by the model.
Focus on Scissors
Lower scores for scissors: This requires additional optimization, including more extensive representation of data on scissors and fine-tuning the model's architecture or hyperparameters.

8. Conclusion
1.Model Strengths:
•	Classifies multi-class data with balanced accuracy.
•	Dropout layers and proper regularization minimize overfitting.
2.Model Limitations:
•	Performance on the scissors class needs improvement; it is probably due to a lack of representation in the data.
•	Dimensionality reduction techniques such as PCA could be used for interpretability.
3.Future Work:
•	Use data augmentation or synthetic data generation to balance class representation.
•	Experiment with different architectures or hyperparameters to optimize performance.

References
•	https://scikit-learn.org/stable/
•	https://keras.io/
•	https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
•	https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
•	Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929-1958.
•	Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.

