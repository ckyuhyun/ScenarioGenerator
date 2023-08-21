import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from helper.data_label import Data_label
import logging
import seaborn as sns
import matplotlib.pyplot as plt


class data_model:
    def __init__(self, seed_data: pd.DataFrame):
        pd.set_option("display.max_columns", None)

        # Data preprocessing
        self.seed_data = seed_data.drop(['ActionRunStatus'], axis=1)
        self.seed_data_columns = self.seed_data.columns
        self.model_data_columns = ['TestScenarioGuid', 'CurrentActionGuid','CategoryGuid', 'ActionMethodType', 'PrevActionGuid', 'PreviousActionCategoryGuid', 'NextActionGuid','NextActionCategoryGuid' ]
        self.label_columns = ['TestScenarioGuid', 'CurrentActionGuid', 'PrevActionGuid','NextActionGuid', 'CategoryGuid', 'PreviousActionCategoryGuid','NextActionCategoryGuid']
        self.model_seed_data = pd.DataFrame(columns=self.model_data_columns)
        self.__data_group()

        # labeling
        self.dl = Data_label(self.model_seed_data)
        self.__post_processing_with_label()

        self.model_seed_data.to_csv('model_seed_data.csv')

    def get_model_seed_data(self):
        if self.model_seed_data.isna().sum().sum() != 0:
            raise Exception("There is empty(null) data")

        return self.model_seed_data

    def get_start_entry(self):
        pass

    def get_original_value_by_label(self, label_group_name, label):
        return self.dl.get_value_of_label(label_group_name=label_group_name, search_label=label)

    def get_current_action_data_by_label(self, label) -> list:
        d = self.model_seed_data.loc[self.model_seed_data['CurrentActionGuid_label'] == label]\
                                .drop(['NextActionGuid_label'], axis=1)
        return d.values

    def get_action_guids_by_stack_index(self, stack_index, unique_data_retured=False):
        """
        return action guids of a specific stack index
        :param stack_index:
        :return:
        """
        action_guids = self.get_action_guid_by_stack_index(stack_index)
        if unique_data_retured:
            unique_list = []
            for d in action_guids:
                if d not in unique_list:
                    unique_list.append(d)
            return unique_list
        else:
            return action_guids

    def get_action_guid_label_by_action_guid(self, action_guid: str):
        """
        return a label of an action guid
        :param action_guid:
        :return:
        """
        return self.dl.get_label_of_value('ActionLabel', action_guid)


    def get_action_guid_by_stack_index(self, stack_index) -> list:
        """
        Collect guid of actions with a specific stack index
        :param stack_index:
        :return: list of action guid
        """
        actions_guid = self.seed_data.loc[self.seed_data['stackIndex'] == stack_index]['TestActionGuid']
        return actions_guid.values



    def __data_group(self):
        _group_data = self.seed_data.sort_values(['ActionStartDate'], ascending=True).groupby('ScenarioHistoryGuid')

        # The gk indicates each key of group
        for gk in list(_group_data.groups.keys()):
            # get a size of a grouping value
            value_length = len(_group_data.get_group(gk).values)
            _group_value = _group_data.get_group(gk)
            _group_value['sequential_index'] = range(0, len(_group_value))

            for v in _group_value.values:
                # get a value of a current action
                val = pd.DataFrame(data=v.reshape((1, len(_group_value.columns))), columns=_group_value.columns)
                # get a stack index of a current action in a current scenario
                current_action_stack_index = val['sequential_index'].values[0]

                # get guid of a previous and next action of the current action
                try:
                    prev_action_guid = _group_value.loc[lambda x: x['sequential_index'] == (current_action_stack_index - 1)]['TestActionGuid'].iloc[0] if current_action_stack_index > 0 else None
                    prev_action_category_guid = _group_value.loc[lambda x: x['sequential_index'] == (current_action_stack_index - 1)]['CategoryGuid'].iloc[0] if current_action_stack_index > 0 else None
                except Exception as e:
                    raise Exception(f"Action Guid : {val['TestActionGuid'].values[0]}")

                try:
                    next_action_guid = None if current_action_stack_index >= (value_length - 1) else _group_value.loc[lambda x: x['sequential_index'] == (current_action_stack_index + 1)]['TestActionGuid'].iloc[0]
                    next_action_category_guid = None if current_action_stack_index >= (value_length - 1) else _group_value.loc[lambda x: x['sequential_index'] == (current_action_stack_index + 1)]['CategoryGuid'].iloc[0]
                except Exception as e:
                    raise Exception(f"Action Guid : {val['TestActionGuid'].values[0]}")

                # Having a new data in dataframe
                new_row_data = pd.DataFrame({'TestScenarioGuid': val['TestScenarioGuid'].values[0],
                                             'CurrentActionGuid': val['TestActionGuid'].values[0],
                                             'CategoryGuid': val['CategoryGuid'].values[0],
                                             'ActionMethodType': val['ActionMethodType'].values[0],
                                             'PrevActionGuid':  val['TestActionGuid'].values[0] if prev_action_guid is None else prev_action_guid,
                                             'PreviousActionCategoryGuid': val['TestActionGuid'].values[0] if prev_action_category_guid is None else prev_action_category_guid,
                                             'NextActionGuid': val['TestActionGuid'].values[0] if next_action_guid is None else next_action_guid,
                                             'NextActionCategoryGuid': val['TestActionGuid'].values[0] if next_action_category_guid is None else next_action_category_guid,
                                             #'ActionRunStatus': val['ActionRunStatus'].values[0],
                                             }, index=[0])

                self.model_seed_data = pd.concat([self.model_seed_data, new_row_data], ignore_index=True)

    def __post_processing_with_label(self):
        """
        Generate labels for column value
        :return:
        """
        # labels for Test scenario
        self.dl.add_group_label_column(group_columns=['TestScenarioGuid'], label_group_name="TestScenarioLabel")
        # labels for Action
        self.dl.add_group_label_column(group_columns=['CurrentActionGuid', 'PrevActionGuid', 'NextActionGuid'], label_group_name="ActionLabel")
        # labels for Category
        self.dl.add_group_label_column(group_columns=['CategoryGuid', 'PreviousActionCategoryGuid', 'NextActionCategoryGuid'], label_group_name="CategoryLabel")
        df = self.dl.get_data()



        # dropping columns having their label column
        self.model_seed_data = df.drop(columns=self.label_columns, axis=1)


