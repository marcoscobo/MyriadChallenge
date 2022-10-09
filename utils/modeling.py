from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class modeling:

    def __init__(self, model, X_train, y_train, X_test):

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.cv_folds = 5
        self.auc_score = None
        self.cv_scores = None
        self.cv_score = None
        self.X_train_t = None
        self.X_test_t = None
        self.y_train_t = None
        self.y_test_t = None
        self.y_pred_t = None
        self.fitted_model = None
        self.y_pred = None

    @staticmethod
    def auc_custom_score(y_true, y_pred):

        return roc_auc_score(y_true, y_pred)

    def cv(self, cv_train=False, verbose=True, plot_roc=True):

        self.auc_score = make_scorer(self.auc_custom_score, greater_is_better=True)

        if cv_train:

            self.X_train_t, self.X_test_t, self.y_train_t, self.y_test_t = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=0)
            self.cv_scores = cross_val_score(self.model, self.X_train_t, self.y_train_t, cv=self.cv_folds, scoring=self.auc_score)
            if verbose:
                print("Mean AUC training sub-training set cross-validation score: %.2f" % self.cv_scores.mean())
                print(self.cv_scores)

            self.fitted_model = self.model.fit(self.X_train_t, self.y_train_t)
            self.y_pred_t = self.fitted_model.predict(self.X_test_t)
            self.cv_score = self.auc_custom_score(self.y_test_t, self.y_pred_t)
            if verbose:
                print("AUC training sub-testing set score: %.2f" % self.cv_score)
            if plot_roc:
                fpr, tpr, thresholds = roc_curve(self.y_test_t, self.y_pred_t)
                plt.figure(figsize=(9,9))
                plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % self.cv_score)
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver operating characteristic example")
                plt.legend(loc="lower right")
                plt.show()

        else:

            self.cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv_folds, scoring=self.auc_score)
            if verbose:
                print("Mean AUC training set cross-validation score: %.2f" % self.cv_scores.mean())
                print(self.cv_scores)

    def fit_data(self):

        self.model = self.model.fit(self.X_train, self.y_train)

    def predict_data(self):

        self.y_pred = self.model.predict(self.X_test)

        return self.y_pred