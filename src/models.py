#imports
from sklearn.tree import *
import threading
import numpy as np
#LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#random forest
from sklearn.ensemble import RandomForestClassifier

class DTModel(threading.Thread):
    """Threaded Decision Tree Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

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
        sd = dtModel.predict(XT)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Decision Tree Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of Decision Tree Model: %.2f" % (acc)+' %')

class RFModel(threading.Thread):
    """Threaded Random Forest Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT = XT
        self.YT = YT
        self.accLabel = accLabel

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
        sd = rfModel.predict(XT)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Random Forest Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of Random Forest Model: %.2f" % (acc)+' %')

class LSTMModel(threading.Thread):
    """Threaded LSTM Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT = XT
        self.YT = YT
        self.accLabel = accLabel

    def run(self):
        X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        XT = self.XT.reshape((self.XT.shape[0], self.XT.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, self.Y, epochs=10, batch_size=32, verbose=0)
        _, acc = model.evaluate(XT, self.YT, verbose=0)
        print("Accuracy of LSTM Model: %.2f" % (acc*100)+" %")
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of LSTM Model: %.2f" % (acc*100)+" %")
