# This code has been modified from the original work by Nagabhushan S Baddi (InStep Intern at Infosys Ltd.)

#import the modules
import dataset_load
import models
import threading
import time
import pickle
import os
import glob

#global variable
#file list hardcoded
files = ['../dataset\\capture20110810.binetflow.pickle', '../dataset\\capture20110811.binetflow.pickle', '../dataset\\capture20110812.binetflow.pickle', '../dataset\\capture20110815-2.binetflow.pickle', '../dataset\\capture20110815-3.binetflow.pickle', '../dataset\\capture20110815.binetflow.pickle', '../dataset\\capture20110816-2.binetflow.pickle', '../dataset\\capture20110816-3.binetflow.pickle', '../dataset\\capture20110816.binetflow.pickle', '../dataset\\capture20110817.binetflow.pickle', '../dataset\\capture20110818-2.binetflow.pickle', '../dataset\\capture20110819.binetflow.pickle']
model_name = ""

def load_data(file_path):
    """Load the data"""
    with open(file_path, 'rb') as file:
        sd = pickle.load(file, encoding='latin1')
    return sd[0], sd[1], sd[2], sd[3]

def loadModel(modelName, X, Y, XT, YT, curr):
    """load the modelName ML model and test the accuracy"""
    # The models.DTModel, models.LSTMModel, and models.RFModel
    if modelName == 'Decision Tree':
        model = models.DTModel(X, Y, XT, YT, curr)
        model.start()
    elif modelName == 'LSTM':
        model = models.LSTMModel(X, Y, XT, YT, curr)
        model.start()
    elif modelName == 'Random Forest':
        model = models.RFModel(X, Y, XT, YT, curr)
        model.start()
    else:
        print("Invalid option selected!")

def next_model(curr, model_name):
    print(curr)
    if(curr < len(files)):
        file_path = files[curr]
        print(f"Processing {file_path}...")
        X, Y, XT, YT = load_data(file_path)
        loadModel(model_name, X, Y, XT, YT, curr)
    else:
        print("all model loaded")

if __name__ == "__main__":

    print("Choose a model to run:")
    print("0: Decision Tree")
    print("1: Random Forest")
    print("2: LSTM")

    directory = "../dataset"
            
    try:
        choice = int(input("Enter your choice (0/1/2): "))
        modelMapping = {
        0: 'Decision Tree',
        1: 'Random Forest',
        2: 'LSTM'
        }

        if choice in modelMapping:
             modelName = modelMapping[choice]
             next_model(0, modelName)
        else:
             print("Invalid choice! Please enter 0, 1, or 2.")
                    
    except ValueError:
        print("Invalid input! Please enter a number.")
