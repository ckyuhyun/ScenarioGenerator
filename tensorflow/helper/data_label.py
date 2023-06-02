from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Data_labelling:
    def __init__(self, data: pd.DataFrame):
        self.__le = LabelEncoder()
        self.__seed_data = data

    def get_data_with_label(self, columns: list) -> pd.DataFrame:

        for _column in columns:
            __unique_data = self.__get_unique_data(_column)
            __label_data = self.__le.fit_transform(__unique_data)
            self.__seed_data[_column].apply(lambda : x )






    def __get_unique_data(self, column):
        data_set = list(self.__seed_data[column])
        unique_data_set = []
        for d in data_set:
            if d not in unique_data_set:
                unique_data_set.append(d)

        return unique_data_set



