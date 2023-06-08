import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from helper.data_label import Data_labelling
import logging


class data_preprocessing:
    def __init__(self, seed_data: pd.DataFrame):
        pd.set_option("display.max_columns", None)
        self.seed_data = seed_data
        self.seed_data_columns = seed_data.columns
        self.model_data_columns =['CurrentActionGuid', 'PrevActionGuid', 'NextActionGuid', 'CategoryGuid', 'ActionRunStatus']
        self.model_seed_data = pd.DataFrame(columns= self.model_data_columns)


        self.__data_group()
        self.__labelling()




    def get_model_seed_data(self):
        return self.model_seed_data


    def get_start_entry(self):
        pass

    def __data_group(self):
        _group_data = self.seed_data.sort_values(['stackIndex'], ascending=True).groupby('ScenarioHistryGuid')

        for gk in list(_group_data.groups.keys()):
            value_length = len(_group_data.get_group(gk).values)
            for v in _group_data.get_group(gk).values:
                # get a value of a current action
                val = pd.DataFrame(data=v.reshape((1, 9)), columns=self.seed_data_columns)
                # get a stack index of a current action in a current scenario
                current_action_stack_index = val['stackIndex'].values[0]

                # get guid of a previous and next action of the current action
                prev_action_guid = _group_data.get_group(gk).loc[lambda x: x['stackIndex'] == (current_action_stack_index - 1)]['TestActionGuid'].iloc[0] if current_action_stack_index > 0 else None

                try:
                    next_action_guid = None if current_action_stack_index >= (value_length-1) else _group_data.get_group(gk).loc[lambda x: x['stackIndex'] == (current_action_stack_index + 1)]['TestActionGuid'].iloc[0]
                except:
                    raise Exception(f"Action Guid : {val['TestActionGuid'].values[0]}")


                new_row_data = pd.DataFrame({'CurrentActionGuid': val['TestActionGuid'].values[0],
                                             'PrevActionGuid': prev_action_guid,
                                             'NextActionGuid': next_action_guid,
                                             'CategoryGuid': val['CategoryGuid'].values[0],
                                             'ActionRunStatus': val['ActionRunStatus'].values[0]}, index=[0])
                self.model_seed_data = pd.concat([self.model_seed_data, new_row_data], ignore_index=True)

        print('')

    def __labelling(self):
        dl = Data_labelling(self.seed_data)
        labelling_columns = ['TestActionGuid', 'TestScenarioGuid', 'Category']
        df = dl.get_data_with_label(columns=labelling_columns)
        labelling_columns.append('CategoryGuid')
        labelling_columns.append('ActionName')
        dropped_df = df.drop(columns=labelling_columns, axis=1)






