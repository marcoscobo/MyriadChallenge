# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class tensorflow_modeling:

    def __init__(self, X_train, y_train, X_test):

        self.model = None
        self.epochs = 100
        self.batch_size = 32
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=False),
        ]
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.history = None
        self.auc_score = None
        self.X_train_t = None
        self.X_test_t = None
        self.y_train_t = None
        self.y_test_t = None
        self.y_pred_t = None
        self.y_pred = None

    @staticmethod
    def make_model(input_shape):

        input_layer = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ReLU()(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)

        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ReLU()(conv3)

        gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = tf.keras.layers.Dense(2, activation="softmax")(gap)

        return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def cv(self, y_to_prob=False, verbose=True, verbose_auc=True, plot_train_val=True, plot_roc=True):

        self.X_train_t, self.X_test_t, self.y_train_t, self.y_test_t = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=0)

        self.X_train_t = np.array(self.X_train_t).reshape((self.X_train_t.shape[0], self.X_train_t.shape[1], 1))
        self.X_test_t = np.array(self.X_test_t).reshape((self.X_test_t.shape[0], self.X_test_t.shape[1], 1))
        if y_to_prob:
            self.y_train_t = tf.one_hot(self.y_train_t, depth=2)

        self.model = self.make_model(input_shape=self.X_train_t.shape[1:])

        self.model.compile(
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_crossentropy']
        )

        self.history = self.model.fit(
            self.X_train_t,
            self.y_train_t,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=0.2,
            verbose=verbose,
        )

        if plot_train_val:
            metric = "sparse_categorical_crossentropy"
            plt.figure(figsize=(16,9))
            plt.plot(self.history.history[metric])
            plt.plot(self.history.history["val_" + metric])
            plt.title("model " + metric)
            plt.ylabel(metric, fontsize="large")
            plt.xlabel("epoch", fontsize="large")
            plt.legend(["train", "val"], loc="best")
            plt.show()

        self.y_pred_t = self.model.predict(self.X_test_t, verbose=verbose)
        self.y_pred_t = pd.Series(self.y_pred_t[:, 1]).apply(lambda x: 1 if x > 0.5 else 0)

        self.auc_score = roc_auc_score(self.y_test_t, self.y_pred_t)

        if verbose_auc:
            print("AUC training sub-testing set score: %.2f" % self.auc_score)
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(self.y_test_t, self.y_pred_t)
            plt.figure(figsize=(9,9))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % self.auc_score)
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right")
            plt.show()

    def fit_data(self, y_to_prob=False, verbose=True, plot_train_val=True):

        self.X_train = np.array(self.X_train).reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.array(self.X_test).reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        if y_to_prob:
            self.y_train = tf.one_hot(self.y_train, depth=2)

        self.model = self.make_model(input_shape=self.X_train.shape[1:])

        self.model.compile(
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_crossentropy']
        )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_split=0.2,
            verbose=verbose,
        )

        if plot_train_val:
            metric = "sparse_categorical_crossentropy"
            plt.figure(figsize=(16,9))
            plt.plot(self.history.history[metric])
            plt.plot(self.history.history["val_" + metric])
            plt.title("model " + metric)
            plt.ylabel(metric, fontsize="large")
            plt.xlabel("epoch", fontsize="large")
            plt.legend(["train", "val"], loc="best")
            plt.show()

    def predict_data(self, verbose=True):

        self.y_pred = self.model.predict(self.X_test, verbose=verbose)
        self.y_pred = pd.Series(self.y_pred[:, 1]).apply(lambda x: 1 if x > 0.5 else 0)

        return self.y_pred
