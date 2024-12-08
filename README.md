# Lithium-Ion-Battery-Sate-of-Health-Estimation
This repository contains my final year capstone project. In this project I worked on improving the state of health estimation in Lithium Ion battery management systems. The solution was implemented using Bi-LSTM and Attention Mechanism. 

## Table of Contents
* Introduction
* Dataset
* Model Training
* Best Model Selection
* Optimized Model
* Technologies Used
* Functionalities
* Conclusion
* Contributors
  
## Introduction

- Customer churn prediction involves building machine learning models to identify customers who are likely to leave the service. In this project, we employ neural networks, a subset of machine learning, to predict customer churn based on historical data.

## Dataset

- The dataset used for this project is the NASA lithium ion battery dataset for determining the remaining useful life. This dataset can be found using this [link](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) The dataset contains information such as state of charge, voltage, current and temperature of the cells in the pack. And these values are used for estimating state of health.

## Model Training

- I trained multiple neural network models with different hyperparameters and used the Cuckoo Search Algorithm for hyperparameter optimization to find the best parameters. The model was first trained using the LSTM then later with Bi-LSTM to find the optimal model. The training is performed on a training set, and model performance is evaluated on a validation set. An Attention Mechanism was used to solve the memory loss effects on long sequential data by the LSTM AND Bi-LSTM.
  
## Best Model Selection

- We use grid search with cross-validation to find the best hyperparameters for our neural network. The model with the highest validation performance is selected as the best model. The evaluation metrics for this model on the test set are reported.

## Optimized Model

- An optimized neural network model is created based on the best hyperparameters. This model is then trained and evaluated to assess its performance. The goal is to further fine-tune the model and achieve better results.

## Technologies Used:
1. Python

    Description: Python serves as the primary programming language for the development of the battery state of health estimation model.

2. Machine Learning Libraries
a. TensorFlow and Keras

    Description: TensorFlow is employed as the deep learning framework, and Keras, integrated with TensorFlow, is used to build and train neural network models.

b. Scikit-learn

    Description: Scikit-learn is utilized for various machine learning tasks, including data preprocessing, model evaluation, and hyperparameter tuning.

3. Data Analysis and Visualization
a. Pandas

   Description: Pandas is used for data manipulation and analysis, particularly for handling the dataset.

b. Matplotlib and Seaborn

    Description: Matplotlib and Seaborn are used for data visualization to gain insights into the dataset and the model performance.

4. Jupyter Notebooks

    Description: Google Colab Notebooks are employed for an interactive and collaborative development environment. The code for this project is organized and presented in google colab Notebooks to enhance readability and understanding.


## Functionality:
- Users can input collect data on the operation of their packs using current, temperature and voltage sensors. This data will be used to make the estimations.


## Conclusion

- The project concludes with the increase in the accuracy of estimating the state of health of lithium ion batteries which are used for energy storage in the smart grid and electric vehicles. 

## Contributors:
-  Tinashe Blackie Kanukai
  
