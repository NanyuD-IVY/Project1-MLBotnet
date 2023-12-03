# This code has been modified from the original work by Nagabhushan S Baddi (InStep Intern at Infosys Ltd.)

#imports
from sklearn.tree import *
import threading
import numpy as np
import main
import os
import json

#compare
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score


#LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#random forest
from sklearn.ensemble import RandomForestClassifier

#excel
import pandas as pd
from openpyxl import load_workbook

#plotting
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#global variable
result_path = 'result.xlsx'

def finish_model_running(curr, model_name):
    main.next_model(curr, model_name)

def print_metrics(YT, sd, y_scores=None, model_name=''):
    tn, fp, fn, tp = confusion_matrix(YT, sd).ravel()
    
    # Calculating metrics
    metrics = {
        'Model': model_name,
        'True Positives (TP)': tp,
        'True Negatives (TN)': tn,
        'False Positives (FP)': fp,
        'False Negatives (FN)': fn,
        'Recall': recall_score(YT, sd),
        'F1 Score': f1_score(YT, sd),
        'Balanced Accuracy': balanced_accuracy_score(YT, sd),
        'Geometric Mean': geometric_mean_score(YT, sd),
        'Matthews Correlation Coefficient': matthews_corrcoef(YT, sd),
        'ROC-AUC Score': None,
        'Precision-Recall AUC': None
    }

    metrics['ROC-AUC Score'] = roc_auc_score(YT, y_scores)
    
    precision, recall, _ = precision_recall_curve(YT, y_scores)
    metrics['Precision-Recall AUC'] = auc(recall, precision)

    # Print the metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Convert the dictionary to a DataFrame
    result_df = pd.DataFrame([metrics])

    # Append to Excel file, creating if it does not exist
    if os.path.exists(result_path):
        with pd.ExcelWriter(result_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            result_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        result_df.to_excel(result_path, index=False)

    fpr, tpr, _ = roc_curve(YT, y_scores)
    roc_auc = roc_auc_score(YT, y_scores)
    # Prepare plotting plotting data for JSON
    new_data = {
        'model_name': model_name,
        'fpr': fpr.tolist(),  # Convert numpy arrays to lists for JSON serialization
        'tpr': tpr.tolist(),
        'roc_auc': roc_auc
    }

    # Read existing data and append new data
    output_file = "plotData\\" + model_name + "_plot.json"
    data_to_append = []
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data_to_append = json.load(file)

    # Append new data under the model name key
    data_to_append.append(new_data)

    # Write updated data to JSON file
    with open(output_file, 'w') as file:
        json.dump(data_to_append, file, indent=4)

class DTModel(threading.Thread):
    """Threaded Decision Tree Model"""
    def __init__(self, X, Y, XT, YT, curr, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel
        self.curr = curr

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        dtModel = DecisionTreeClassifier()
        dtModel.fit(X, Y)
        
        # Predictions for metrics
        sd = dtModel.predict(XT)
        y_scores = dtModel.predict_proba(XT)[:, 1]  # Probability of positive class

        print_metrics(YT, sd, y_scores, 'decision tree')
        
        finish_model_running(self.curr + 1, "Decision Tree")


class RFModel(threading.Thread):
    """Threaded Random Forest Model"""
    def __init__(self, X, Y, XT, YT, curr, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT = XT
        self.YT = YT
        self.accLabel = accLabel
        self.curr = curr

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)

        rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
        rfModel.fit(X, Y)
        
        # Predictions for metrics
        sd = rfModel.predict(XT)
        y_scores = rfModel.predict_proba(XT)[:, 1]  # Probability of positive class

        print_metrics(YT, sd, y_scores, 'rf')
        finish_model_running(self.curr + 1, "Random Forest")


class LSTMModel(threading.Thread):
    """Threaded LSTM Model"""
    def __init__(self, X, Y, XT, YT, curr, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT = XT
        self.YT = YT
        self.accLabel = accLabel
        self.curr = curr

    def run(self):
        X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        XT = self.XT.reshape((self.XT.shape[0], self.XT.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, self.Y, epochs=10, batch_size=32, verbose=0)

        # Getting probabilistic outputs for metrics
        y_scores = model.predict(XT, verbose = 0).flatten()
        predictions = (y_scores > 0.5).astype("int32")

        print_metrics(self.YT, predictions, y_scores, 'lstm')
        finish_model_running(self.curr + 1, "LSTM")
