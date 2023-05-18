import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import logging


class data_preprocessing:
    def __init__(self):
        pd.set_option("display.max_columns", None)
        self.__unique_action_id_collections = []
        self.__unique_action_page_collections = []
        self.__unique_action_name_collections = []
        self.__unique_scenario_id_collections = []
        self.__unique_scenario_action_id_collections = []

        _action_distance_fileName = "../action_distance.csv"
        _action_page_fileName = "../action_data.csv"

        # private properties
        self.__action_distance_data = pd.read_csv(_action_distance_fileName, index_col=[0])
        self.__action_page_data = pd.read_csv(_action_page_fileName, index_col=[0])
        self.__action_distance_data.columns = [c.strip() for c in self.__action_distance_data.columns]
        self.__action_page_data.columns = [c.strip() for c in self.__action_page_data.columns]

        self.__action_distance_label_data = []
        self.__action_page_label_data = []
        self.__merged_data = None # data of after getting columns dropped
        self.__backup_merged_data = None  # data of before getting columns dropped
        self.__start_entry_collection = None

        # Feature label
        self.__df_action_id_category = []
        self.__df_action_page_category = []

    def get_src_data(self):
        return self.__action_distance_data

    def get_compiled_data(self):
        return self.__merged_data

    def init(self):

        self.__generate_unique_collection()

        # generate the label based on the seed data
        self.__data_merge()
        self.__run_action_distance_labelling()
        self.__run_action_data_labelling()
        self.__add_action_page_label()
        self.__add_action_name_label()
        # self.__add_scenario_id_label()
        # #self.__add_scenario_action_label()

        self.data_wrapup()
        self.__merged_data.to_csv("merged_data.csv")

        print('')

    def __data_merge(self):
        action_id_page = []
        action_id_name = []
        next_action_page = []
        next_action_name = []
        merged_data = pd.DataFrame(columns=['ActionGuid', 'Page', 'ActionName'])

        # update matched data for each row
        for action_id, next_id in zip(self.__action_distance_data['ActionId'], self.__action_distance_data['Next']):
            # looking up the matched page and action name for each row
            matched_action_data = self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0]
            action_id_page.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0]['Page'])
            action_id_name.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == action_id].iloc[0]['ActionName'])
            next_action_page.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == next_id].iloc[0]['Page'])
            next_action_name.append(self.__action_page_data.loc[self.__action_page_data['ActionGuid'].values == next_id].iloc[0]['ActionName'])

            if len(merged_data['ActionGuid']) == 0 or matched_action_data['ActionGuid'] not in merged_data['ActionGuid'].values:
                df_matched_data = pd.DataFrame([matched_action_data], columns=['ActionGuid', 'Page', 'ActionName'])
                merged_data = pd.concat([merged_data, df_matched_data], ignore_index=True)

        self.__action_distance_data['ActionIdPage'] = action_id_page
        self.__action_distance_data['ActionIdName'] = action_id_name
        self.__action_distance_data['NextPage'] = next_action_page
        self.__action_distance_data['NextActionName'] = next_action_name

        # reordering of columns
        self.__merged_data = pd.DataFrame(columns=["ActionId", "Action_id_label", "ActionIdPage", "ActionIdPageLabel", "ActionIdName", "ActionName_Label", "Next", "Next_id_label", "NextPage", "NextPageLabel",
                                                    "NextActionName", "NextActionName_Label", "distance", "rating"])

        for column in self.__action_distance_data.columns:
            self.__merged_data[column] = self.__action_distance_data[column].values

        # updated_columns = df_reordered.columns.tolist()
        # updated_columns.insert(updated_columns.index('ActionId') + 1, 'ActionIdPage')
        # updated_columns.insert(updated_columns.index('ActionIdPage') + 1, 'ActionIdName')
        # updated_columns.insert(updated_columns.index('Next') + 1, 'NextPage')
        # updated_columns.insert(updated_columns.index('NextPage') + 1, 'NextActionName')
        # #self.merged_data = self.data.loc[:, ["ActionGuid", "ActionIdPage", "ActionIdName", "Next", "NextPage", "NextActionName", "distance", "rating"]]
        # self.merged_data = self.action_distance_data.loc[:, updated_columns]

    def data_wrapup(self):
        self.__backup_merged_data = self.__merged_data
        self.__merged_data = self.__merged_data.drop(columns=['ActionId', 'ActionIdPage', 'ActionIdName', 'Next', 'NextPage', 'NextActionName'])
        self.collect_start_entry()

    def collect_start_entry(self):
        start_entry_collection = []

        for unique_scenario_id in self.__unique_scenario_id_collections:
            start_entry_collection.append(self.__action_page_data.groupby("ScenarioGuid").get_group(unique_scenario_id)["ActionGuid"].iloc[0])

        self.__start_entry_collection = start_entry_collection

    def __add_action_page_label(self):
        # Generate label for current page and next action
        le = LabelEncoder()
        df_page_collection = pd.DataFrame(data=self.__unique_action_page_collections, columns=["Page"])
        action_page_label = le.fit_transform(self.__unique_action_page_collections)
        df_page_collection["Page_Label"] = pd.DataFrame(data=action_page_label)

        # Generate a new column with rows for page label matching with a page name
        action_id_page_label_column = []
        self.__merged_data["ActionIdPage"].apply(lambda x: action_id_page_label_column.append(df_page_collection.loc[df_page_collection['Page'] == x]['Page_Label'].values[0]))
        self.__merged_data["ActionIdPageLabel"] = action_id_page_label_column

        next_action_id_page_label_column = []
        self.__merged_data["NextPage"].apply(lambda x: next_action_id_page_label_column.append(
            df_page_collection.loc[df_page_collection['Page'] == x]['Page_Label'].values[0]))
        self.__merged_data["NextPageLabel"] = next_action_id_page_label_column

        self.__merged_data.columns = [c.strip() for c in self.__merged_data.columns]

    def __add_action_name_label(self):
        # Generate label for current and next action name
        le = LabelEncoder()

        # Action Name label
        df_action_name_collection = pd.DataFrame(data=self.__unique_action_name_collections, columns=["ActionName"])
        action_name_label = le.fit_transform(self.__unique_action_name_collections)
        df_action_name_collection['ActionName_Label'] = pd.DataFrame(data=action_name_label)

        current_action_name_label_column = []
        self.__merged_data["ActionIdName"].apply(lambda x: current_action_name_label_column.append(
            df_action_name_collection.loc[df_action_name_collection['ActionName'] == x]['ActionName_Label'].values[0]))
        self.__merged_data["ActionName_Label"] = current_action_name_label_column

        next_action_name_label_column = []
        self.__merged_data["NextActionName"].apply(lambda x: next_action_name_label_column.append(
            df_action_name_collection.loc[df_action_name_collection['ActionName'] == x]['ActionName_Label'].values[0]))
        self.__merged_data["NextActionName_Label"] = next_action_name_label_column

        self.__merged_data.columns = [c.strip() for c in self.__merged_data.columns]

    def __add_scenario_id_label(self):
        le = LabelEncoder()

        scenario_id_label = le.fit_transform(self.__unique_scenario_id_collections)
        scenario_id_collection = pd.DataFrame(data=self.__unique_action_name_collections, columns=["ScenarioGuid"])
        scenario_id_collection['ScenarioGuidLabel'] = pd.DataFrame(data=scenario_id_label)

        scenario_ids_label_column = []
        self.__merged_data["ScenarioGuid"].apply(lambda x: scenario_ids_label_column.append(
            scenario_id_collection.loc[scenario_id_collection["ScenarioGuid"] == x]['ScenarioGuidLabel']))

        self.__merged_data['ScenarioGuidLabel'] = scenario_ids_label_column

    def __add_scenario_action_label(self):
        le = LabelEncoder()

    def get_page_by_action_id(self, action_id):
        return self.__action_page_data.loc[self.__action_page_data['ActionGuid'] == action_id].iloc[0]['ActionName']

    def get_page_of_action_id_label(self, action_id_label):
        pass

    def get_start_entry(self):
        return random.choice(self.__start_entry_collection)

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

        return self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category'] == action_id].iloc[0]['Action_Id_Category_label']

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
        return self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category_label'] == action_id_label].iloc[0]['Action_Id_Category']

    def get_next_action(self, current_action_id):
        neighbour_nodes = []

        _labelled_neighbour_nodes = self.__action_distance_data.loc[self.__action_distance_data['ActionId'] == current_action_id]['Next']

        if len(_labelled_neighbour_nodes) == 0:
            return -1, neighbour_nodes

        for _n_nodes in _labelled_neighbour_nodes:
            neighbour_nodes.append(_n_nodes)
        try:
            return random.choice(list(neighbour_nodes)), neighbour_nodes
        except Exception as e:
            raise Exception(f"No Action Id category label {str(e)}, {str(len(neighbour_nodes))}")

    def get_current_neighbor_nodes(self, current_action_id):
        _labelled_neighbour_nodes = self.__action_distance_data.loc[self.__action_distance_data['ActionId'] == current_action_id]['Next']

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

        # labelling with action ids
        self.__df_action_id_category = pd.DataFrame(pd.Series(self.__unique_action_id_collections), columns=['Action_Id_Category'])
        self.__df_action_id_category['Action_Id_Category_label'] = le.fit_transform(self.__df_action_id_category['Action_Id_Category'])
        self.__action_distance_label_data = self.__action_distance_data[['ActionId', 'Next', 'distance', 'rating']].copy()

        for x, y in zip(self.__df_action_id_category['Action_Id_Category'].values,
                        self.__df_action_id_category['Action_Id_Category_label'].values):
            self.__action_distance_label_data['ActionId'] = self.__action_distance_label_data['ActionId'].replace([x], y)
            self.__action_distance_label_data['Next'] = self.__action_distance_label_data['Next'].replace([x], y)

        # Add columns for label of Action ids
        current_action_ids_label_column = []
        next_ids_label_column = []
        for _current_id, _next_id in zip(self.__merged_data[['ActionId']].values, self.__merged_data[['Next']].values):
            try:
                current_action_ids_label_column.append(self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category'].values == _current_id].iloc[0]['Action_Id_Category_label'])
                next_ids_label_column.append(self.__df_action_id_category.loc[self.__df_action_id_category['Action_Id_Category'].values == _next_id].iloc[0]['Action_Id_Category_label'])
            except Exception as e:
                logging.error("fail to collect data :" + str(e))

        self.__merged_data['Action_id_label'] = current_action_ids_label_column
        self.__merged_data['Next_id_label'] = next_ids_label_column

        print('')

        '''
        # Generate dummy columns for the Next
        labelled_dummy_df = pd.get_dummies(data=self.__action_distance_label_data, columns=['Next'])
        # labelled_dummy_df.to_csv("data_dummy.csv", sep=',')
        labelled_dummy_df_next_columns = [c for c in labelled_dummy_df.columns if "Next" in c]

        for column in labelled_dummy_df_next_columns:
            labelled_dummy_df[column] = labelled_dummy_df[column].astype(int)

        start_entry = labelled_dummy_df.iloc[0]['ActionId']
        '''



    def __run_action_data_labelling(self):
        le = LabelEncoder()
        self.__action_page_label_data = []
        self.__df_action_page_category = pd.DataFrame(pd.Series(self.__unique_action_page_collections), columns=['action_page_category'])
        self.__df_action_page_category['action_page_category_label'] = le.fit_transform(self.__df_action_page_category['action_page_category'])

    def __generate_unique_collection(self):
        # collect unique action Ids
        self.__generate_unique_action_ids()
        # collect unique action page
        self.__generate_unique_action_page()
        # collect unique action page
        self.__generate_unique_action_name()
        # collect unique scenario id
        self.__generate_unique_scenario_ids()
        # collect unique scenario action id
        # self.__generate_unique_scenario_action_ids()

    def __generate_unique_action_ids(self):
        action_id_collections = list(self.__action_distance_data['ActionId'].values) + list(self.__action_distance_data['Next'].values)
        for a in action_id_collections:
            if a not in self.__unique_action_id_collections:
                self.__unique_action_id_collections.append(a)

    def __generate_unique_action_page(self):
        action_page_name_collections = list(self.__action_page_data['Page'])
        for p in action_page_name_collections:
            if p not in self.__unique_action_page_collections:
                self.__unique_action_page_collections.append(p)

    def __generate_unique_action_name(self):
        action_name_collections = list(self.__action_page_data['ActionName'])
        for p in action_name_collections:
            if p not in self.__unique_action_name_collections:
                self.__unique_action_name_collections.append(p)

    def __generate_unique_scenario_ids(self):
        scenario_id_collections = list(self.__action_page_data['ScenarioGuid'])
        for p in scenario_id_collections:
            if p not in self.__unique_scenario_id_collections:
                self.__unique_scenario_id_collections.append(p)

    def __generate_unique_scenario_action_ids(self):
        scenario_action_id_collections = list(self.__action_page_data['ScenarioActionGuid'])
        for p in scenario_action_id_collections:
            if p not in self.__unique_scenario_action_id_collections:
                self.__unique_scenario_action_id_collections(p)
