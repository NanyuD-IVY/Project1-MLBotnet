# This code has been modified from the original work by Nagabhushan S Baddi (InStep Intern at Infosys Ltd.)

#import the modules
import dataset_load
import models
import threading
import time
import pickle
import os

def load_data(file_path):
    """Load the data"""
    file = open('../dataset/flowdata.pickle', 'rb')
    sd = pickle.load(file, encoding='latin1')
    return sd[0], sd[1], sd[2], sd[3]

def loadModel(modelName, X, Y, XT, YT):
    """load the modelName ML model and test the accuracy"""
    if modelName == 'Decision Tree':
        model = models.DTModel(X, Y, XT, YT)
        model.start()
    elif modelName == 'LSTM':
        model = models.LSTMModel(X, Y, XT, YT)
        model.start()
    elif modelName == 'Random Forest':
        model = models.RFModel(X, Y, XT, YT)
        model.start()
    else:
        print("Invalid option selected!")

#text prompt for the file name and choice of model
if __name__ == "__main__":
    file_path = input("Please enter the path of your binetflow file: ")

    if os.path.exists(file_path):
        X, Y, XT, YT = load_data(file_path)
    else:
        print("The file path does not exist. Please check and try again.")
        exit()

    print("Choose a model to run:")
    print("0: Decision Tree")
    print("1: Random Forest")
    print("2: LSTM")
    
    try:
        choice = int(input("Enter your choice (0/1/2): "))
        modelMapping = {
            0: 'Decision Tree',
            1: 'Random Forest',
            2: 'LSTM'
        }

        if choice in modelMapping:
            modelName = modelMapping[choice]
            loadModel(modelName, X, Y, XT, YT)
        else:
            print("Invalid choice! Please enter 0, 1, or 2.")
            
    except ValueError:
        print("Invalid input! Please enter a number.")
