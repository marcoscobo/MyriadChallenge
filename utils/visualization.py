from pandas import DataFrame

class visualization:

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.n_plots = 1
        self.n_lines_plot = 10

    @staticmethod
    def get_X_series(X_train):

        X_series = (X_train + 1).cumprod(axis=1).T
        X_series = X_series - X_series.iloc[0]

        return X_series

    def plot_series(self):

        for i in range(self.n_plots):
            X_plot = self.get_X_series(self.X_train.iloc[self.n_lines_plot * i: self.n_lines_plot * (i + 1)])
            colors = self.y_train.iloc[self.n_lines_plot * i: self.n_lines_plot * (i + 1)].apply(lambda x: 'green' if x == 1 else 'red')

            plot = X_plot.plot(figsize=(16,9), color=colors, legend=False)

    def plot_density(self):

        for i in range(self.n_plots):
            X_plot = self.X_train.iloc[self.n_lines_plot * i: self.n_lines_plot * (i + 1)].T
            colors = self.y_train.iloc[self.n_lines_plot * i: self.n_lines_plot * (i + 1)].apply(lambda x: 'green' if x == 1 else 'red')

            plot = X_plot.plot(kind='density', figsize=(16,9), color=colors, legend=False, xlim=(-0.2,0.2))

    def plot_mean_density(self):

        real_mean = self.X_train[self.y_train == 1].mean()
        syn_mean = self.X_train[self.y_train == 0].mean()

        plot = DataFrame({'real':real_mean, 'syn':syn_mean}).plot(kind='density', figsize=(16,9), color=['green', 'red'])