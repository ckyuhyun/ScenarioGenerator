import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from helper.data_label import Data_labelling
import logging



class data_preprocessing:
    def __init__(self, seed_data: pd.DataFrame):
        pd.set_option("display.max_columns", None)
        self.seed_data = seed_data.drop(['ActionRunStatus'], axis=1)
        self.seed_data_columns = self.seed_data.columns
        self.model_data_columns = ['TestScenarioGuid', 'CurrentActionGuid', 'PrevActionGuid', 'NextActionGuid', 'CategoryGuid', 'ActionMethodType']
        self.label_columns = ['TestScenarioGuid', 'CurrentActionGuid', 'PrevActionGuid', 'NextActionGuid', 'CategoryGuid']
        self.model_seed_data = pd.DataFrame(columns=self.model_data_columns)

        self.__data_group()
        self.__post_processing_with_label()

    def get_model_seed_data(self):
        return self.model_seed_data

    def get_start_entry(self):
        pass

    def __data_group(self):
        _group_data = self.seed_data.sort_values(['stackIndex'], ascending=True).groupby('ScenarioHistryGuid')

        column_num = len(self.seed_data.columns)

        for gk in list(_group_data.groups.keys()):
            value_length = len(_group_data.get_group(gk).values)
            for v in _group_data.get_group(gk).values:
                # get a value of a current action
                val = pd.DataFrame(data=v.reshape((1, column_num)), columns=self.seed_data_columns)
                # get a stack index of a current action in a current scenario
                current_action_stack_index = val['stackIndex'].values[0]

                # get guid of a previous and next action of the current action
                prev_action_guid = _group_data.get_group(gk).loc[lambda x: x['stackIndex'] == (current_action_stack_index - 1)]['TestActionGuid'].iloc[0] if current_action_stack_index > 0 else ''

                try:
                    next_action_guid = '' if current_action_stack_index >= (value_length - 1) else _group_data.get_group(gk).loc[lambda x: x['stackIndex'] == (current_action_stack_index + 1)]['TestActionGuid'].iloc[0]
                except Exception as e:
                    raise Exception(f"Action Guid : {val['TestActionGuid'].values[0]}")

                # Having a new data in dataframe
                new_row_data = pd.DataFrame({'TestScenarioGuid': val['TestScenarioGuid'].values[0],
                                             'CurrentActionGuid': val['TestActionGuid'].values[0],
                                             'PrevActionGuid': prev_action_guid,
                                             'NextActionGuid': next_action_guid,
                                             'CategoryGuid': val['CategoryGuid'].values[0],
                                             #'ActionRunStatus': val['ActionRunStatus'].values[0],
                                             'ActionMethodType': val['ActionMethodType'].values[0]
                                             }, index=[0])
                self.model_seed_data = pd.concat([self.model_seed_data, new_row_data], ignore_index=True)

    def __post_processing_with_label(self):
        dl = Data_labelling(self.model_seed_data)
        df = dl.get_data_with_label(columns=self.label_columns)

        # dropping columns having their label column
        self.model_seed_data = df.drop(columns=self.label_columns, axis=1)
