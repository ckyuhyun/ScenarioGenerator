from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Data_label:
    def __init__(self, data: pd.DataFrame):
        self.__le = LabelEncoder()
        self.seed_data = data
        self.label_collection = {}

    def add_group_label_column(self, group_columns: list, label_group_name:str) -> pd.DataFrame:
        """
        This is generating the same labels across the columns passed along
        :param group_columns:
        :param label_group_name:
        :return:
        """
        _unique_label_data = self._get_group_column_label(group_columns, label_group_name)
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

    def     get_value_of_label(self, label_group_name: str, search_label):
        return self.label_collection[label_group_name].get(search_label)


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

    def update_label_lookup_table(self, label_group_name:str, original_value:list, label_value:list):
        column_labels_dic = {}
        for ov, lv in zip(original_value, label_value):
            column_labels_dic[lv] = ov

        self.label_collection[label_group_name] = column_labels_dic



    def _get_group_column_label(self, group_columns: list, label_group_name:str) -> pd.DataFrame:
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

        self.update_label_lookup_table(label_group_name, _unique_data['data'].values.tolist(), _unique_data['data_label'].values.tolist())

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




