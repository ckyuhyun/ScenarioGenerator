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
        self.__data = pd.read_csv(_action_distance_fileName)
        self.__action_page_data = pd.read_csv(_action_page_fileName)
        self.__data.columns = [c.strip() for c in self.__data.columns]
        self.__action_page_data.columns = [c.strip() for c in self.__action_page_data.columns]

        self.__data = self.__data.loc[:, ["ActionId", "Next", "distance", "rating"]]
        self.__action_page_data = self.__action_page_data.loc[:, ["ActionGuid", "Page", "ActionName"]]
        self.__action_distance_label_data = []
        self.__labelled_action_page_data = []
        self.merged_data = None

        # public properties
        self.__start_entry_id = None

        # Feature label
        self.__df_action_id_category = []
        self.__df_action_page_category = []

    def get_src_data(self):
        return self.__data

    def init(self):

        self.__generate_unique_collection()

        # generate the label based on the seed data
        self.__run_action_distance_labelling()
        self.__run_action_data_labelling()

        self.__data_merge()
        self.__add_action_page_label()
        self.__add_action_name_label()

        print('')

    def __data_merge(self):
        action_id_page = []
        action_id_name = []
        next_action_page = []
        next_action_name = []

        #_action_detail_data = self.__action_page_data[["ActionGuid", "Page", "ActionName"]]

        #self.data.columns = ["ActionId", "ActionIdPage", "ActionIdName", "Next", "NextPage", "NextActionName", "distance", "rating"]

        merged_data = pd.DataFrame(columns=['ActionGuid', 'Page', 'ActionName'])

        # update matched data for each row
        for action_id, next_id in zip(self.__data['ActionId'], self.__data['Next']):
            # looking up the relevant action for reference data source
            matched_action_data = self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0]
            action_id_page.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0]['Page'])
            action_id_name.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0][
                'ActionName'])
            next_action_page.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == next_id].iloc[0][
                'Page'])
            next_action_name.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == next_id].iloc[0][
                'ActionName'])

            if len(merged_data['ActionGuid']) == 0 or matched_action_data['ActionGuid'] not in merged_data['ActionGuid'].values:
                df_matched_data = pd.DataFrame([matched_action_data], columns=['ActionGuid', 'Page', 'ActionName'])
                merged_data = pd.concat([merged_data,df_matched_data], ignore_index=True)

        self.__data['ActionIdPage'] = action_id_page
        self.__data['ActionIdName'] = action_id_name
        self.__data['NextPage'] = next_action_page
        self.__data['NextActionName'] = next_action_name

        # reordering of columns
        self.merged_data = self.__data.loc[:, ["ActionId", "ActionIdPage", "ActionIdName", "Next", "NextPage", "NextActionName", "distance", "rating"]]

    def __add_action_page_label(self):
        # Generate label for current page and next action
        le = LabelEncoder()
        df_page_collection = pd.DataFrame(data=self.unique_action_page_collections, columns=["Page"])
        action_page_label = le.fit_transform(self.unique_action_page_collections)
        df_page_collection["Page_Label"] = pd.DataFrame(data=action_page_label)

        # Generate a new column with rows for page label matching with a page name
        action_id_page_label_column = []
        self.merged_data["ActionIdPage"].apply(lambda x : action_id_page_label_column.append(df_page_collection.loc[df_page_collection['Page'] == x]['Page_Label'].values[0]))
        self.merged_data["ActionIdPageLabel"] = action_id_page_label_column

        next_action_id_page_label_column = []
        self.merged_data["NextPage"].apply(lambda x: next_action_id_page_label_column.append(
            df_page_collection.loc[df_page_collection['Page'] == x]['Page_Label'].values[0]))
        self.merged_data["NextPageLabel"] = next_action_id_page_label_column

        self.merged_data.columns = [c.strip() for c in self.merged_data.columns]

    def __add_action_name_label(self):
        # Generate label for current and next action name
        le = LabelEncoder()

        df_action_name_collection = pd.DataFrame(data=self.unique_action_name_collections, columns=["ActionName"])
        action_name_label = le.fit_transform(self.unique_action_name_collections)
        df_action_name_collection['ActionName_Label'] = pd.DataFrame(data=action_name_label)

        action_name_label_column = []
        self.merged_data["NextActionName"].apply(lambda x: action_name_label_column.append(
            df_action_name_collection.loc[df_action_name_collection['ActionName'] == x]['ActionName_Label'].values[0]))

        self.merged_data["NextActionName_Label"] = action_name_label_column

        self.merged_data.columns = [c.strip() for c in self.merged_data.columns]



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
        _labelled_neighbour_nodes = self.__data.loc[self.__data['ActionId'] == current_action_id]['Next']

        neighbour_nodes = []
        for _n_nodes in _labelled_neighbour_nodes:
            neighbour_nodes.append(_n_nodes)

        return random.choice(list(neighbour_nodes)), neighbour_nodes

    def get_current_neighbor_nodes(self, current_action_id):
        _labelled_neighbour_nodes = self.__data.loc[self.__data['ActionId'] == current_action_id]['Next']

        neighbour_nodes = []
        for _n_nodes in _labelled_neighbour_nodes:
            neighbour_nodes.append(_n_nodes)

        return neighbour_nodes

    def __run_action_distance_labelling(self):
        """
        Generate label for action Ids
        :return:
        """
        le = LabelEncoder()
        self.__action_distance_label_data = []
        self.__df_action_id_category = pd.DataFrame(pd.Series(self.unique_action_id_collections),
                                                    columns=['Action_Id_Category'])
        self.__df_action_id_category['Action_Id_Category_label'] = le.fit_transform(
            self.__df_action_id_category['Action_Id_Category'])

        self.__action_distance_label_data = self.__data[['ActionId', 'Next', 'distance', 'rating']].copy()

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
        le = LabelEncoder()
        self.__labelled_action_page_data = []

        self.__df_action_page_category = pd.DataFrame(pd.Series(self.unique_action_page_collections),
                                                      columns=['action_page_category'])
        self.__df_action_page_category['action_page_category_label'] = le.fit_transform(
            self.__df_action_page_category['action_page_category'])

    def __generate_unique_collection(self):
        # collect unique action Ids
        self.__generate_unique_action_ids()
        # collect unique action page
        self.__generate_unique_action_page()
        # collect unique action page
        self.__generate_unique_action_name()

    def __generate_unique_action_ids(self):
        action_id_collections = list(self.__data['ActionId'].values) + list(self.__data['Next'].values)
        for a in action_id_collections:
            if a not in self.unique_action_id_collections:
                self.unique_action_id_collections.append(a)

    def __generate_unique_action_page(self):
        action_page_name_collections = list(self.__action_page_data['Page'])
        for p in action_page_name_collections:
            if p not in self.unique_action_page_collections:
                self.unique_action_page_collections.append(p)

    def __generate_unique_action_name(self):
        action_name_collections = list(self.__action_page_data['ActionName'])
        for p in action_name_collections:
            if p not in self.unique_action_name_collections:
                self.unique_action_name_collections.append(p)
