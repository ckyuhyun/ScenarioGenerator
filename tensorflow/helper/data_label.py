from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Data_labelling:
    def __init__(self, data: pd.DataFrame):
        self.__le = LabelEncoder()
        self.__seed_data = data

    def get_data_with_label(self, columns: list) -> pd.DataFrame:

        for _column in columns:
            _unique_label_data = self._get_unique_label_data(_column)
            _column_label_data = []
            # assign a label to each value in the column
            self.__seed_data[_column].apply(lambda x: _column_label_data.append(_unique_label_data.loc[_unique_label_data['data'] == x].iloc[0]['data_label']))

            self.__seed_data[_column+'_label'] = _column_label_data

    def _get_unique_label_data(self, column:str) -> pd.DataFrame:
        _unique_data = pd.DataFrame(self.__get_unique_data(column), columns=["data"])
        _unique_data['data_label'] = self.__le.fit_transform(_unique_data)

        return _unique_data

    def __get_unique_data(self, column :str):
        data_set = list(self.__seed_data[column])
        unique_data_set = []
        for d in data_set:
            if d not in unique_data_set:
                unique_data_set.append(d)

        return unique_data_set



