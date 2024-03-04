from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import numpy as np
import sys

class AggData():
    def __init__(self, root):
        self.root = root

    def get_lowlevel_list(self):
        lowlevel_task_paths = []
        for root, dir, files in os.walk(self.root):
            for file in files:
                file_path = os.path.join(root,file)
                if file.endswith('lowlevel.csv'):
                    lowlevel_task_paths.append(file_path)
        print("\n1",lowlevel_task_paths)
        return lowlevel_task_paths

    def extract_label(self,file_npy):
        return np.load(file_npy,allow_pickle = True).item()

    def agg_all_df(self):
        """
        given path to root folder, returns an aggregated df with lowlevel features and all labels
        """
        lowlevel_task_paths = self.get_lowlevel_list()
        df_all = []
        for path in sorted(lowlevel_task_paths):
            df = pd.read_csv(path)
            task_num = path.split('/')[-1].split('_')[1]
            aiim_num = path.split('/')[2][4:] 
            df['task'] = task_num
            df['aiim'] = aiim_num
            df['valence'] = self.extract_label(f'{self.root}/aiim{aiim_num}/task_{task_num}_valence.npy')
            df['arousal'] = self.extract_label(f'{self.root}/aiim{aiim_num}/task_{task_num}_arousal.npy')
            df['mental_demand']= self.extract_label(f'{self.root}/aiim{aiim_num}/task_{task_num}_mental_demand.npy')
            df_all.append(df)
        
        df_all = pd.concat(df_all, ignore_index=True)
        #df_all.to_csv("./agg_data.csv")
        return df_all
    

class FeatureImportance():
    def __init__(self,df_aiim):
        self.df = df_aiim
        self.X = None
        self.y = None

    def clean_df(self):
        self.X = self.df.drop(['task','aiim','valence','arousal','mental_demand'], axis=1) 
        self.X = self.X.replace([np.nan,np.inf, -np.inf], 0)
        self.y = self.df['mental_demand']
    
    def rank_features(self):
        self.clean_df()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        importances = model.feature_importances_
        features = self.X.columns

        return importances, features


def compare_feature_importance_by_aiim(df_all):
    """
    ranks the feature importance by aiim (person), and creates a plot that compares them
    """
    grouped = df_all.groupby('aiim')
    fig, ax = plt.subplots(1, len(grouped), figsize=(6.4,7), sharey=True)
    
    df_all_importance = None
    for i, (name, group) in enumerate(grouped):
        fi = FeatureImportance(group)
        importance, features = fi.rank_features()
        df_fi = pd.DataFrame({'feature': features, f'importance_{name}': importance})
        df_fi = df_fi.sort_values(by=f'importance_{name}', ascending=False).reset_index(drop=True)

        if i==0:
            df_all_importance = df_fi
        else:
            df_all_importance = pd.merge(df_all_importance,df_fi,on='feature')

        ax[i].barh(df_fi['feature'], df_fi[f'importance_{name}'])
        ax[i].set_title(f'Feature Importances for {name}')

    plt.tight_layout()
    plt.savefig('compare_feature_importance.png')
    df_all_importance.to_csv('./feature_importance.csv')


def main():
    ad = AggData(sys.argv[1])  #"../sample_dataset"
    df_all = ad.agg_all_df()
    compare_feature_importance_by_aiim(df_all)


if __name__ == "__main__":
    main()