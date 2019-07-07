# -*-coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
import keras
from keras.utils import plot_model
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt 


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        
        plt.title('Model accuracy')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        plt.savefig('acc.png')
        plt.show()
        
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.title('Model loss')        
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig('loss.png')
        plt.show()

def CNNTrain(X, y):
    """
    训练CNN模型，将模型保存，并返回model
    :param X: X
    :param y: y
    :return: model
    """
    X /= 255
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(26, 18, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    # model.add(Conv2D(108, (3, 3), activation='rule'))
    # model.add(Conv2D(108, (3, 3), activation='rule'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(36, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = LossHistory()

    model.fit(X, y,
              batch_size=1000,
              epochs=20,
              verbose=1,
              validation_split=0.2,
              callbacks=[history])
    history.loss_plot('epoch')

    return model

if __name__ == '__main__':
    X = joblib.load('../Data/TrainArray/X.pkl')
    y = joblib.load('../Data/TrainArray/y.pkl')
    model = CNNTrain(X, y)
    joblib.dump(model, r'../Model/model/cnn2.pkl', protocol=2)
    
    plot_model(model, to_file='model.png', show_shapes=True)
    
    
