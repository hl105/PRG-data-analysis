import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
from numpy.lib.stride_tricks import sliding_window_view


class RollingWindowPandas():
    def __init__(self,filenum,option,bot_name):
        self.bot_name = bot_name
        self.filenum = filenum
        self.option = option
        self.df = pd.read_csv(f"./sample_dataset/{self.bot_name}/task_{self.filenum}_lowlevel.csv")
    
    def exponential_smoothing(self):
        """
        weighted moving average technique,smoothing is done by 
        assigning exponentially decreasing weights to the past observations.
        """
        expSmoothed = pd.DataFrame()
        for col in self.df.columns:
            try:
                model = SimpleExpSmoothing(self.df[col], initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
                expSmoothed[col] = model.fittedvalues
            except ConvergenceWarning:
                print(f"convergence warning for column {col}")
            continue
        
        expSmoothed.to_csv(f'./smoothed_data/{self.bot_name}/task_{i}_lowlevel_expSmoothed.csv')
        return expSmoothed
    
    def rolling_window_analysis(self):
        rolling_stats = {}

        for col in self.df.columns: 
            rolling_stats[col] = self.df[col]
            rolling_obj = self.df[col].rolling(window=self.option)
            rolling_stats[f"{col}_rolling_avg"] = rolling_obj.mean()
            rolling_stats[f"{col}_rolling_median"] = rolling_obj.median()
            rolling_stats[f"{col}_rolling_std"] = rolling_obj.std()
            rolling_stats[f"{col}_rolling_min"] = rolling_obj.min()
            rolling_stats[f"{col}_rolling_max"] = rolling_obj.max()

        df_rolling = pd.DataFrame(rolling_stats)

        df_rolling.to_csv(f'./smoothed_data/{self.bot_name}/task_{i}_lowlevel_rolling_5.csv')
        return df_rolling
    
    def plot_smoothed(self,df_smoothed): 
        """
        Line chart depicting the expSmoothed / moving average (pick1) and the actual values
        option = 0: expSmoothed
        option = 1: rolling_avg
        """
        plt.figure(figsize=(20, 48 * 4)) 

        for i, col in enumerate(self.df.columns):
            ax = plt.subplot(48, 1, i + 1)
            ax.plot(self.df[col], marker="o", color="black", label='Original')
            if self.option == -1:
                ax.plot(df_smoothed[col], marker="o", color="blue", label='Smoothed')
            else:
                ax.plot(df_smoothed[f"{col}_rolling_avg"], marker="o", color="green", label='5 window avg')
            ax.set_title(col)
        
        if self.option == -1:
            plt.savefig(f'./smoothed_data/{self.bot_name}/task_{self.filenum}_lowlevel_expSmoothed.jpg')
        else:
            plt.savefig(f'./smoothed_data/{self.bot_name}/task_{self.filenum}_lowlevel_rolling_5.jpg')
    
    def choose_smoothing_method(self):
        if self.option == -1:
            df_smoothed  = self.exponential_smoothing()
            self.plot_smoothed(df_smoothed)
        else:
            df_smoothed = self.rolling_window_analysis()
            self.plot_smoothed(df_smoothed)

class AggRollingWindow:

    def __init__(self,filenum,window_size,bot_name):
        self.bot_name = bot_name
        self.window_size = window_size
        self.data = np.load(f"./sample_dataset/{self.bot_name}/task_{filenum}_highlevel.npy")
        self.sliding_window = sliding_window_view(self.data, self.window_size)
    
    def rolling_window_mean(self):
        """
        return convolution of data & seq of ones (whose length = window)
        """
        return np.convolve(self.data, np.ones(self.window_size), 'valid') / self.window_size # same vs valid? boundary effects may be seen 

    def rolling_window_median(self):
        return np.median(self.sliding_window,axis=1)

    def rolling_window_min(self):
        return self.sliding_window.min(axis=1)

    def rolling_window_max(self):
        return self.sliding_window.max(axis=1)

    def match_data_size(self, data_altered):
        return np.resize(data_altered, self.data.shape[0])*(self.data/self.data)
    
    def add_features_per_window(self):
        """
        creates numpy array with columns as: original,mean,median,max,min
        """
        data_mean = self.match_data_size(self.rolling_window_mean())
    
        data_median = self.match_data_size(self.rolling_window_median())
        data_min = self.match_data_size(self.rolling_window_min()) 
        data_max = self.match_data_size(self.rolling_window_max())

        data_agg_features = np.column_stack((self.data,data_mean, data_median, data_min, data_max))
        np.save(f"./smoothed_data/{self.bot_name}/task_{i}_highlevel_rolling.npy",data_agg_features)
        return data_agg_features
    

if __name__ == '__main__':

    #tried using Pandas for .csv files (lowlevel)
    for bot in ["aiim001","aiim002"]:
        for i in range(1,10):
            for j in [-1,3,5]: #-1 means we do exponential smoothing, not rolling window
                data_smoothed = RollingWindowPandas(i,j,bot)
                print(f"{bot}: finished task {i} with option {j}")
                data_smoothed.choose_smoothing_method()

    #tried using NumPy for .npy files (highlevel)
    for bot in ["aiim001","aiim002"]:
        for i in range(1,10): #for task 1-9
            for j in [3,5]: # for rolling window size 3,5
                data_rolling = AggRollingWindow(i,j,bot)
                data_agg_features = data_rolling.add_features_per_window()
                print(f"{bot}: check first row in task {i} with window {j}: {data_agg_features[0,:]}")
        
    
    
    
    