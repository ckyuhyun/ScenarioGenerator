import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from helper.data_label import Data_labelling
import logging


class data_preprocessing:
    def __init__(self, seed_data):
        pd.set_option("display.max_columns", None)
        self.seed_data = seed_data

        self.__labelling()

    def __labelling(self):
        dl = Data_labelling(self.seed_data)
        labelling_columns = ['TestActionGuid', 'TestScenarioGuid', 'Category']
        df = dl.get_data_with_label(columns=labelling_columns)
        labelling_columns.append('CategoryGuid')
        labelling_columns.append('ActionName')
        dropped_df = df.drop(columns=labelling_columns, axis=1)






