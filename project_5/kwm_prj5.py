"""

"""

import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat




def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    class_names = ['CryptMT', 'Dragon', 'HC', 'NLS', 'Rabbit', 'Salsa20', 'Sosemanuk', 'LEX']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # Import Data
    x_tr = loadmat('PerfWeb/Training_data_40classes_Tor_Browser.mat')['X_train3']
    y_tr = loadmat('PerfWeb/Training_label_40classes_Tor_Browser.mat')['Ytrain'].T
    x_val = loadmat('PerfWeb/Test_data_40classes_Tor_Browser.mat')['X_test']
    y_val = loadmat('PerfWeb/Test_label_40classes_Tor_Browser.mat')['Ytest'].T
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_val.shape)
    print(y_val.shape)

    # # SVM
    # print('############### SVM ###############')
    # # define model hyperparameters
    # kernel = ['linear','poly','rbf','sigmoid']
    # C = [0.01,0.1,1.] # regularization parameter
    # # all combinations of hyperparameters
    # H = np.array(np.meshgrid(kernel, C)).T.reshape(-1, 2)
    # # to find best performing hyperparams h*, initialize minimum val loss
    # fCE_star = 0  # highest fce accuracy
    # model_star = None  # weights trained by best hyperparam set
    # j = 0  # current iteration in hyperparam set loop
    # for h in H:
    #     # define this training sessions hyperparameters
    #     kern = h[0]
    #     C = float(h[1])
    #     # define and train model
    #     clf = svm.SVC(C=C, kernel=kern)
    #     clf.fit(x_tr, y_tr)
    #     val_preds = clf.predict(x_val)
    #     val_acc = np.sum(val_preds == y_val.flatten()) / float(len(y_val))
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
    #     # check to see if this is the best performing hyperparam set
    #     if val_acc > fCE_star:
    #         print('new best!')
    #         fCE_star = val_acc
    #         model_star = clf
    #
    # # KNN
    # print('############### KNN ###############')
    # # define model hyperparameters
    # n_neigh = [1,5,20,100]  # K for KNN
    # weights = ['uniform', 'distance']  # regularization parameter
    # # all combinations of hyperparameters
    # H = np.array(np.meshgrid(n_neigh, weights)).T.reshape(-1, 2)
    # # to find best performing hyperparams h*, initialize minimum val loss
    # fCE_star = 0 # lowest final loss obtained by a hyperparam set
    # model_star = None  # weights trained by best hyperparam set
    # j = 0  # current iteration in hyperparam set loop
    # for h in H:
    #     # define this training sessions hyperparameters
    #     n_n = int(h[0])
    #     weights = h[1]
    #     # define and train model
    #     neigh = KNeighborsClassifier(n_neighbors=n_n, weights=weights, algorithm='auto')
    #     neigh.fit(x_tr, y_tr)
    #     val_preds = neigh.predict(x_val)
    #     val_acc = np.sum(val_preds == y_val.flatten()) / float(len(y_val))
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
    #     # check to see if this is the best performing hyperparam set
    #     if val_acc > fCE_star:
    #         print('new best!')
    #         fCE_star = val_acc
    #         model_star = neigh
    #
    # # DT
    # print('############### Decision Tree ###############')
    # # define model hyperparameters
    # splitter = ['best', 'random']
    # criterion = ['entropy', 'gini']  # split criterion
    # # all combinations of hyperparameters
    # H = np.array(np.meshgrid(splitter, criterion)).T.reshape(-1, 2)
    # # to find best performing hyperparams h*, initialize minimum val loss
    # fCE_star = 0  # lowest final loss obtained by a hyperparam set
    # model_star = None  # weights trained by best hyperparam set
    # j = 0  # current iteration in hyperparam set loop
    # for h in H:
    #     # define this training sessions hyperparameters
    #     splitter = h[0]
    #     criterion = h[1]
    #     # define and train model
    #     dt = tree.DecisionTreeClassifier(splitter=splitter, criterion=criterion)
    #     dt.fit(x_tr, y_tr)
    #     val_preds = dt.predict(x_val)
    #     val_acc = np.sum(val_preds == y_val.flatten()) / float(len(y_val))
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
    #     # check to see if this is the best performing hyperparam set
    #     if val_acc > fCE_star:
    #         print('new best!')
    #         fCE_star = val_acc
    #         model_star = dt
    #

    # CNN
    # convert to one hot encoding
    nb_classes = 40
    targets = y_tr.reshape(-1) - 1
    y_tr = np.eye(nb_classes)[targets]
    targets = y_val.reshape(-1) - 1
    y_val = np.eye(nb_classes)[targets]

    # print('############### CNN ###############')
    # # define model hyperparameters
    # layers = [2,5]
    # conv_pool_size = [2,4]
    # filters = [8,64]
    # bs = [10]  # batch size
    # dropout = [0., 0.5]
    # # all combinations of hyperparameters
    # H = np.array(np.meshgrid(layers, conv_pool_size, filters, bs, dropout)).T.reshape(-1, 5)
    # # to find best performing hyperparams h*, initialize minimum val loss
    # fCE_star = 0 # lowest final loss obtained by a hyperparam set
    # model_star = None  # weights trained by best hyperparam set
    # j = 0  # current iteration in hyperparam set loop
    # for h in H:
    #     # define this training sessions hyperparameters
    #     l = int(h[0])
    #     c_p_size = int(h[1])
    #     f = int(h[2])
    #     bs = int(h[3])
    #     d = float(h[4])
    #     # define and train model
    #     model = keras.Sequential()
    #     model.add(keras.layers.Conv2D(filters=f, kernel_size=c_p_size, padding='same', activation='relu',
    #                                   input_shape=(x_tr.shape[1], 1, 1)))
    #     model.add(keras.layers.MaxPooling2D(pool_size=(c_p_size, 1)))
    #     model.add(keras.layers.Dropout(d))
    #     for _ in range(l-1):
    #         model.add(keras.layers.Conv2D(filters=f, kernel_size=c_p_size, padding='same', activation='relu'))
    #         model.add(keras.layers.MaxPooling2D(pool_size=(c_p_size, 1)))
    #         model.add(keras.layers.Dropout(d))
    #     model.add(keras.layers.Flatten())
    #     model.add(keras.layers.Dense(40, activation='softmax'))
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     history = model.fit(np.expand_dims(np.expand_dims(x_tr, axis=-1), axis=-1), y_tr,
    #                         validation_data=(np.expand_dims(np.expand_dims(x_val, axis=-1), axis=-1), y_val),
    #                         verbose=0, batch_size=bs, epochs=50)
    #     val_loss = history.history['val_acc'][-1]
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation acc for h ', h, ': ', val_loss)
    #     # check to see if this is the best performing hyperparam set
    #     if val_loss > fCE_star:
    #         print('new best!')
    #         fCE_star = val_loss
    #         model_star = model

    # LSTM RNN
    print('############### LSTM RNN ###############')
    # define model hyperparameters
    layers = [0]  # number of layers
    n_n = [16, 64]  # number of neurons in a layer
    bs = [10]  # batch size
    dropout = [0.]
    # all combinations of hyperparameters
    H = np.array(np.meshgrid(layers, n_n, bs, dropout)).T.reshape(-1, 4)
    # to find best performing hyperparams h*, initialize minimum val loss
    fCE_star = 0  # lowest final loss obtained by a hyperparam set
    model_star = None  # weights trained by best hyperparam set
    j = 0  # current iteration in hyperparam set loop
    for h in H:
        # define this training sessions hyperparameters
        l = int(h[0])
        nn = int(h[1])
        bs = int(h[2])
        d = float(h[3])
        # define and train model
        model = keras.Sequential()
        model.add(LSTM(nn, return_sequences=True, batch_input_shape=(bs, x_tr.shape[1], 1),
                       stateful=True))
        model.add(keras.layers.Dropout(d))
        for _ in range(l):
            model.add(LSTM(nn, return_sequences=True, stateful=True))
        model.add(LSTM(nn, return_sequences=False, stateful=True))
        model.add(Dense(40, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(np.expand_dims(x_tr, axis=-1), y_tr,
                            validation_data=(np.expand_dims(x_val, axis=-1), y_val),
                            batch_size=bs, epochs=5, verbose=0)
        val_loss = history.history['val_acc'][-1]
        j += 1  # update counter
        print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_loss)
        # check to see if this is the best performing hyperparam set
        if val_loss > fCE_star:
            print('new best!')
            fCE_star = val_loss
            model_star = model



















