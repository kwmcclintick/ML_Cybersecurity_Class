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
    data = np.loadtxt('PRNG-Dataset/64-bit/train_64bit.txt', delimiter=',')
    x_tr = data[:,0:32]
    y_tr = data[:,-1].astype(np.int32) - 1
    data_val = np.loadtxt('PRNG-Dataset/64-bit/val_64bit.txt', delimiter=',')
    x_val = data_val[:, 0:32]
    y_val = data_val[:, -1].astype(np.int32) - 1
    # data_test = np.loadtxt('PRNG-Dataset/32-bit/test_32bit.txt', delimiter=',')
    # x_t = data_test[:, 0:32]
    # y_t = data_test[:, -1].astype(np.int32) - 1
    #
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
    #     val_acc = np.sum(val_preds == y_val) / float(len(y_val))
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
    #     # check to see if this is the best performing hyperparam set
    #     if val_acc > fCE_star:
    #         print('new best!')
    #         fCE_star = val_acc
    #         model_star = clf
    # svm_preds = model_star.predict(x_t)
    # print('Test accuracy: ', np.sum(svm_preds==y_t)/float(len(y_t)))
    # plot_confusion_matrix(confusion_matrix(y_t, svm_preds), title='SVM')
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
    #     val_acc = np.sum(val_preds == y_val) / float(len(y_val))
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
    #     # check to see if this is the best performing hyperparam set
    #     if val_acc > fCE_star:
    #         print('new best!')
    #         fCE_star = val_acc
    #         model_star = neigh
    # knn_preds = model_star.predict(x_t)
    # print('Test accuracy: ', np.sum(knn_preds == y_t) / float(len(y_t)))
    # plot_confusion_matrix(confusion_matrix(y_t, knn_preds), title='KNN')

    # DT
    print('############### Decision Tree ###############')
    # define model hyperparameters
    splitter = ['best', 'random']
    criterion = ['entropy', 'gini']  # split criterion
    # all combinations of hyperparameters
    H = np.array(np.meshgrid(splitter, criterion)).T.reshape(-1, 2)
    # to find best performing hyperparams h*, initialize minimum val loss
    fCE_star = 0  # lowest final loss obtained by a hyperparam set
    model_star = None  # weights trained by best hyperparam set
    j = 0  # current iteration in hyperparam set loop
    for h in H:
        # define this training sessions hyperparameters
        splitter = h[0]
        criterion = h[1]
        # define and train model
        dt = tree.DecisionTreeClassifier(splitter=splitter, criterion=criterion)
        dt.fit(x_tr, y_tr)
        val_preds = dt.predict(x_val)
        val_acc = np.sum(val_preds == y_val) / float(len(y_val))
        j += 1  # update counter
        print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_acc)
        # check to see if this is the best performing hyperparam set
        if val_acc > fCE_star:
            print('new best!')
            fCE_star = val_acc
            model_star = dt
    # dt_preds = model_star.predict(x_t)
    # print('Test accuracy: ', np.sum(dt_preds == y_t) / float(len(y_t)))
    # plot_confusion_matrix(confusion_matrix(y_t, dt_preds), title='DT')
    dt_preds = model_star.predict(x_val)
    print('Test accuracy: ', np.sum(dt_preds == y_val) / float(len(y_val)))
    plot_confusion_matrix(confusion_matrix(y_val, dt_preds), title='DT')


    # CNN
    # convert to one hot encoding
    nb_classes = 8
    targets = y_tr.reshape(-1)
    y_tr = np.eye(nb_classes)[targets]
    targets = y_val.reshape(-1)
    y_val = np.eye(nb_classes)[targets]
    targets = y_t.reshape(-1)
    y_t = np.eye(nb_classes)[targets]


    # print('############### CNN ###############')
    # # define model hyperparameters
    # layers = [1,5]
    # conv_pool_size = [1,2]
    # filters = [8,64]
    # bs = [10,100]  # batch size
    # dropout = [0.,0.5]
    # # all combinations of hyperparameters
    # H = np.array(np.meshgrid(layers, conv_pool_size, filters, bs, dropout)).T.reshape(-1, 5)
    # # to find best performing hyperparams h*, initialize minimum val loss
    # fCE_star = np.infty  # lowest final loss obtained by a hyperparam set
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
    #     for _ in range(l):
    #         model.add(keras.layers.Conv2D(filters=f, kernel_size=c_p_size, padding='same', activation='relu',
    #                                       input_shape=(x_tr.shape[1], 1, 1)))
    #         model.add(keras.layers.MaxPooling2D(pool_size=(c_p_size, 1)))
    #         model.add(keras.layers.Dropout(d))
    #     model.add(keras.layers.Flatten())
    #     model.add(keras.layers.Dense(8, activation='softmax'))
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     history = model.fit(np.expand_dims(np.expand_dims(x_tr, axis=-1), axis=-1), y_tr,
    #                         validation_data=(np.expand_dims(np.expand_dims(x_val, axis=-1), axis=-1), y_val),
    #                         verbose=0, batch_size=bs, epochs=5)
    #     val_loss = history.history['val_loss'][-1]
    #     j += 1  # update counter
    #     print('(', j, '/', len(H), ')', 'validation CE loss for h ', h, ': ', val_loss)
    #     # check to see if this is the best performing hyperparam set
    #     if val_loss < fCE_star:
    #         print('new best!')
    #         fCE_star = val_loss
    #         model_star = model
    # score = model_star.evaluate(np.expand_dims(np.expand_dims(x_t, axis=-1), axis=-1), y_t, verbose=2)  # Print test accuracy
    # print('\n', 'Test accuracy:', score[1])
    # cnn_preds = model_star.predict(np.expand_dims(np.expand_dims(x_t, axis=-1), axis=-1))
    # preds = np.zeros_like(cnn_preds)
    # preds[np.arange(len(cnn_preds)), cnn_preds.argmax(1)] = 1
    # preds = [np.where(r == 1)[0][0] for r in preds]
    # y_true = [np.where(r==1)[0][0] for r in y_t]
    # plot_confusion_matrix(confusion_matrix(y_true, preds), title='CNN')






    # LSTM RNN
    print('############### LSTM RNN ###############')
    # define model hyperparameters
    layers = [0, 1]  # number of layers
    n_n = [2, 8]  # number of neurons in a layer
    bs = [10]  # batch size
    dropout = [0., 0.5]
    # all combinations of hyperparameters
    H = np.array(np.meshgrid(layers, n_n, bs, dropout)).T.reshape(-1, 4)
    # to find best performing hyperparams h*, initialize minimum val loss
    fCE_star = np.infty  # lowest final loss obtained by a hyperparam set
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
        model.add(LSTM(nn, return_sequences=True, batch_input_shape=(bs, 32, 1),
                       stateful=True))
        model.add(keras.layers.Dropout(d))
        for _ in range(l):
            model.add(LSTM(nn, return_sequences=True, stateful=True))
        model.add(LSTM(nn, return_sequences=False, stateful=True))
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(np.expand_dims(x_tr, axis=-1), y_tr,
                            validation_data=(np.expand_dims(x_val, axis=-1), y_val),
                            batch_size=bs, epochs=5, verbose=0)
        val_loss = history.history['val_loss'][-1]
        j += 1  # update counter
        print('(', j, '/', len(H), ')', 'validation CE acc for h ', h, ': ', val_loss)
        # check to see if this is the best performing hyperparam set
        if val_loss < fCE_star:
            print('new best!')
            fCE_star = val_loss
            model_star = model

    score = model_star.evaluate(np.expand_dims(x_t, axis=-1), y_t, batch_size=10, verbose=0)  # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    lstm_preds = model_star.predict(np.expand_dims(x_t, axis=-1), batch_size=10)
    preds = np.zeros_like(lstm_preds)
    preds[np.arange(len(lstm_preds)), lstm_preds.argmax(1)] = 1
    preds = [np.where(r == 1)[0][0] for r in preds]
    y_true = [np.where(r==1)[0][0] for r in y_t]
    plot_confusion_matrix(confusion_matrix(y_true, preds), title='LSTM RNN')



















