import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


class data_preprocessing:
    def __init__(self):
        pd.set_option("display.max_columns", None)
        self.unique_action_id_collections = []
        self.unique_action_page_collections = []
        self.unique_action_name_collections = []

        _action_distance_fileName = "../action_distance.csv"
        _action_page_fileName = "../action_data.csv"

        # private properties
        self.data = pd.read_csv(_action_distance_fileName)
        self.action_page_data = pd.read_csv(_action_page_fileName)
        self.data.columns = [c.strip() for c in self.data.columns]
        self.action_page_data.columns = [c.strip() for c in self.action_page_data.columns]

        self.data = self.data.loc[:, ["ActionId", "Next", "distance", "rating"]]
        self.action_page_data = self.action_page_data.loc[:, ["ActionGuid", "Page", "ActionName"]]
        self.__action_distance_label_data = []
        self.__labelled_action_page_data = []

        # public properties
        self.__start_entry_id = None

        # Feature label
        self.__le = LabelEncoder()
        self.__df_action_id_category = []
        self.__df_action_page_category = []

    def get_src_data(self):
        return self.data

    def init(self):

        self.__generate_unique_collection()

        # generate the label based on the seed data
        self.__run_action_distance_labelling()
        self.__run_action_data_labelling()

        self.__data_merge()

    def __data_merge(self):
        #_action_detail_data = self.__action_page_data[["ActionGuid", "Page", "ActionName"]]
        merged_data = pd.DataFrame(columns=['ActionGuid', 'Page', 'ActionName'])

        for _d in self.data['ActionId']:
            matched_data = self.action_page_data.loc[self.action_page_data['ActionGuid'].values == _d].iloc[0]
            if len(merged_data['ActionGuid']) == 0 or matched_data['ActionGuid'] not in merged_data['ActionGuid'].values:
                df_matched_data = pd.DataFrame([matched_data], columns=['ActionGuid', 'Page', 'ActionName'])
                merged_data = pd.concat([merged_data,df_matched_data], ignore_index=True)









    def get_page_by_action_id(self, action_id):
        return self.unique_action_page_collections.loc[self.unique_action_page_collections['ActionGuid'] == action_id].iloc[0]['Page']

    def get_page_of_action_id_label(self, action_id_label):
        pass

    def get_start_entry(self):
        return self.__start_entry_id

    def get_label_from_action_id(self, action_id):
        """
        return label of matched action id
        :param action_id: Action Id
        :return: label of passed action id
        """
        if self.__df_action_id_category['Action_Id_Category'] is None:
            raise Exception("No Action Id category")

        if self.__df_action_id_category['Action_Id_Category_label'] is None:
            raise Exception("No Action Id category label")

        return \
        self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category'] == action_id].iloc[0][
            'Action_Id_Category_label']

    def get_action_id_from_label(self, action_id_label):
        """
        return action Id of matched label
        :param action_id_label: label of action
        :return:
        """
        if self.__df_action_id_category['Action_Id_Category'] is None:
            raise Exception("No Action Id category")

        if self.__df_action_id_category['Action_Id_Category_label'] is None:
            raise Exception("No Action Id category label")
        return self.__df_action_id_category.loc[
            self.__df_action_id_category['Action_Id_Category_label'] == action_id_label].iloc[0][
            'Action_Id_Category']

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

    def __run_action_distance_labelling(self):
        """
        Generate label for action Ids
        :return:
        """
        self.__action_distance_label_data = []
        self.__df_action_id_category = pd.DataFrame(pd.Series(self.unique_action_id_collections),
                                                    columns=['Action_Id_Category'])
        self.__df_action_id_category['Action_Id_Category_label'] = self.__le.fit_transform(
            self.__df_action_id_category['Action_Id_Category'])

        self.__action_distance_label_data = self.data[['ActionId', 'Next', 'distance', 'rating']].copy()

        for x, y in zip(self.__df_action_id_category['Action_Id_Category'].values,
                        self.__df_action_id_category['Action_Id_Category_label'].values):
            self.__action_distance_label_data['ActionId'] = self.__action_distance_label_data['ActionId'].replace([x],
                                                                                                                  y)
            self.__action_distance_label_data['Next'] = self.__action_distance_label_data['Next'].replace([x], y)

        # Generate dummy columns for the Next
        labelled_dummy_df = pd.get_dummies(data=self.__action_distance_label_data, columns=['Next'])
        # labelled_dummy_df.to_csv("data_dummy.csv", sep=',')
        labelled_dummy_df_next_columns = [c for c in labelled_dummy_df.columns if "Next" in c]

        for column in labelled_dummy_df_next_columns:
            labelled_dummy_df[column] = labelled_dummy_df[column].astype(int)

        start_entry = labelled_dummy_df.iloc[0]['ActionId']
        self.__start_entry_id = \
            self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category_label'] == start_entry][
                'Action_Id_Category'].iloc[0]

    def __run_action_data_labelling(self):
        self.__labelled_action_page_data = []

        self.__df_action_page_category = pd.DataFrame(pd.Series(self.unique_action_page_collections),
                                                      columns=['action_page_category'])
        self.__df_action_page_category['action_page_category_label'] = self.__le.fit_transform(
            self.__df_action_page_category['action_page_category'])

    def __generate_unique_collection(self):
        # collect unique action Ids
        self.__generate_unique_action_ids()
        # collect unique action page
        self.__generate_unique_action_page()
        # collect unique action page
        self.__generate_unique_action_name()

    def __generate_unique_action_ids(self):
        action_id_collections = list(self.data['ActionId'].values) + list(self.data['Next'].values)
        for a in action_id_collections:
            if a not in self.unique_action_id_collections:
                self.unique_action_id_collections.append(a)

    def __generate_unique_action_page(self):
        action_page_name_collections = list(self.action_page_data['Page'])
        for p in action_page_name_collections:
            if p not in self.unique_action_page_collections:
                self.unique_action_page_collections.append(p)

    def __generate_unique_action_name(self):
        action_name_collections = list(self.action_page_data['ActionName'])
        for p in action_name_collections:
            if p not in self.unique_action_name_collections:
                self.unique_action_name_collections.append(p)
