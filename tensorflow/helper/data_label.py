from sklearn.preprocessing import LabelEncoder
import pandas as pd
class Data_labelling:
    def __init__(self, data: pd.DataFrame):
        self.__le = LabelEncoder()
        self.seed_data = data

    def get_group_label(self, group_columns: list) -> pd.DataFrame:
        """
        This is generating labels based on all values of the passed columns
        :param group_columns:
        :return:
        """
        _unique_label_data = self._get_group_column_label(group_columns)
        for _column in group_columns:
            _column_label_data = []
            self.seed_data[_column].apply(lambda x: _column_label_data.append(_unique_label_data.loc[_unique_label_data['data'] == x].iloc[0]['data_label']))
            self.seed_data[_column + '_label'] = _column_label_data


    def get_single_label(self, columns: list) -> pd.DataFrame:
        """
        This is generating labels based on values of each column differently
        :param columns:
        :return:
        """
        for _column in columns:
            _unique_label_data = self._get_single_column_label(_column)
            _column_label_data = []
            # assign a label to each value in the column
            self.seed_data[_column].apply(lambda x: _column_label_data.append(_unique_label_data.loc[_unique_label_data['data'] == x].iloc[0]['data_label']))

            self.seed_data[_column + '_label'] = _column_label_data


    def get_data(self):
        return self.seed_data

    def _get_single_column_label(self, column:str) -> pd.DataFrame:
        """
        This generates labels for specific columns.
        :param column:
        :return:
        """
        column_values = self.seed_data[column].values
        _unique_data = pd.DataFrame(self.__get_unique_data(column_values), columns=["data"])
        _unique_data['data_label'] = self.__le.fit_transform(_unique_data)

        return _unique_data

    def _get_group_column_label(self, group_columns: list) -> pd.DataFrame:
        """
        This generates a uniform labels for columns grouped.
        :param values:
        :return:
        """
        group_df_values = []
        for column in group_columns:
            group_df_values.extend(list(self.seed_data[column].values))

        _unique_data = pd.DataFrame(data=self.__get_unique_data(group_df_values), columns=['data'])
        _unique_data['data_label'] = self.__le.fit_transform(_unique_data)

        return _unique_data


    def __get_unique_data(self, column :str):
        data_set = list(self.seed_data[column])
        unique_data_set = []
        for d in data_set:
            if d not in unique_data_set:
                unique_data_set.append(d)

        return unique_data_set

    def __get_unique_data(self, values: list):
        unique_data_set = []
        for v in values:
            if v not in unique_data_set:
                unique_data_set.append(v)

        return unique_data_set




