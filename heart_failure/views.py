from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def index(request):
    return render(request, 'Heart_form\\heart_home.html')

def app(request):
    return render(request, 'Heart_form\\form.html')

def add(request):

    dataset = pd.read_csv("heart_failure\\heart_failure_clinical_records_dataset.csv")
    dataset = dataset.drop(0)

#Split the dataset into train and test

# Split the dataset into the features and the target
    features = dataset.iloc[:, [0, 4, 7, 8]]

    target = dataset.iloc[:, 12]
# Create a logistic regression model
    model = LogisticRegression(penalty="l2", C=0.01)

# Fit the model to the data
    model.fit(features, target)
    
    Name = request.GET['Name']
    Age = int(request.GET['Age'])
    Serum_Sodium = int(request.GET['Serum_Sodium'])
    Ejection_Fraction = int(request.GET['Ejection_Fraction'])
    Serum_creatinine = float(request.GET['Serum_creatinine'])
    
    
    values = [Age, Ejection_Fraction, Serum_creatinine, Serum_Sodium]
    df_values = pd.DataFrame(values).T
    df_values.columns = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
    
    rese = model.predict(df_values)
    
    res = rese[0]
    
    if res == 1:
        return render(request, 'heart_form\\heart_results_bad.html')
    else:
        return render(request, 'heart_form\\heart_results_good.html')

def results(request):
    return render(request, 'heart_form\\heart_results_good.html')