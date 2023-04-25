import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


class data_preprocessing:
    def __init__(self):
        self.unique_action_id_collections = []
        _action_distance_fileName = "../action_distance.csv"
        _action_data_fileName = "../action_data.csv"

        # private properties
        self.data = pd.read_csv(_action_distance_fileName)
        self.__action_page_data = pd.read_csv(_action_data_fileName)
        self.data.columns = [c.strip() for c in self.data.columns]
        self.data = self.data.loc[:, ["ActionId", "Next", "distance", "rating"]]
        self.labelled_data = []

        # public properties
        self.start_entry_id = None


        # Feature label
        self.__le = LabelEncoder()
        self.df_category = []

    def get_src_data(self):
        return self.data
    def data_merge(self):
        # pd.set_option("display.max_columns", None)

        _action_detail_data = self.__action_page_data[["ActionGuid", "Page", "ActionName"]]

        # collect unique action Ids
        action_id_collections = list(self.data['ActionId'].values) + list(self.data['Next'].values)
        for a in action_id_collections:
            if a not in self.unique_action_id_collections:
                self.unique_action_id_collections.append(a)

        # genearate the label based on the seed data
        self.__run_label()

    def __run_label(self):
        self.labelled_data = []
        self.df_category = pd.DataFrame(pd.Series(self.unique_action_id_collections), columns=['Action_Id_Category'])
        self.df_category['Action_Id_Category_label'] = self.__le.fit_transform(self.df_category['Action_Id_Category'])

        self.labelled_data = self.data[['ActionId','Next', 'distance','rating']].copy()

        for x, y in zip(self.df_category['Action_Id_Category'].values, self.df_category['Action_Id_Category_label'].values):
            self.labelled_data['ActionId'] = self.labelled_data['ActionId'].replace([x], y)
            self.labelled_data['Next'] = self.labelled_data['Next'].replace([x], y)

        # Generate dummy columns for the Next
        labelled_dummy_df = pd.get_dummies(data=self.labelled_data, columns=['Next'])
        #labelled_dummy_df.to_csv("data_dummy.csv", sep=',')
        labelled_dummy_df_next_columns = [c for c in labelled_dummy_df.columns if "Next" in c]

        for column in labelled_dummy_df_next_columns:
            labelled_dummy_df[column] = labelled_dummy_df[column].astype(int)

        start_entry = labelled_dummy_df.iloc[0]['ActionId']
        self.start_entry_id = self.df_category.loc[self.df_category['Action_Id_Category_label'] == start_entry]['Action_Id_Category'].iloc[0]



    def get_label_from_action_id(self, action_id):
        """
        return label of matched action id
        :param action_id: Action Id
        :return: label of passed action id
        """
        if self.df_category['Action_Id_Category'] is None:
            raise Exception("No Action Id category")

        if self.df_category['Action_Id_Category_label'] is None:
            raise Exception("No Action Id category label")

        return self.df_category.loc[self.df_category['Action_Id_Category'] == action_id].iloc[0]['Action_Id_Category_label']

    def get_action_id_from_label(self, action_id_label):
        """
        return action Id of matched label
        :param action_id_label: label of action
        :return:
        """
        if self.df_category['Action_Id_Category'] is None:
            raise Exception("No Action Id category")

        if self.df_category['Action_Id_Category_label'] is None:
            raise Exception("No Action Id category label")
        return self.df_category.loc[self.df_category['Action_Id_Category_label'] == action_id_label].iloc[0]['Action_Id_Category']

    def get_next_action(self, current_action_id):
        _labelled_neighbour_nodes = self.data.loc[self.data['ActionId'] == current_action_id]['Next']

        neighbour_nodes = []
        for _n_nodes in _labelled_neighbour_nodes:
            neighbour_nodes.append(_n_nodes)

        return random.choice(list(neighbour_nodes)), neighbour_nodes

    def get_current_neighbor_nodes(self, current_action_id):
        _labelled_neighbour_nodes = self.data.loc[self.data['ActionId'] == current_action_id]['Next']

        neighbour_nodes = []
        for _n_nodes in _labelled_neighbour_nodes:
            neighbour_nodes.append(_n_nodes)

        return neighbour_nodes




