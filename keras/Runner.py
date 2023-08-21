import random

import pandas as pd

from Tools.GraphLibrary import ClassificationGraphy
from numpy.core.defchararray import strip
from data_model import data_model
from knn_model import KNN_model
from helper.db_context_helper import dbContext_helper
from Utils.read_yaml_file import  read_yaml
import global_instance

def run():
    model = KNN_model()
    db_context = dbContext_helper()

    # get data
    read_data = pd.read_csv('../extended_action_data.csv', index_col=0)
    data_handler = data_model(read_data)

    # train a knn model
    knn_seed_data = data_handler.get_model_seed_data()
    model.get_seed_data(knn_seed_data)
    model.Run()

    # find an action where it starts with
    action_guids = data_handler.get_action_guids_by_stack_index(0, unique_data_retured=True)
    random_action_guid = random.choice(action_guids)
    current_action_guid_label = data_handler.get_action_guid_label_by_action_guid(random_action_guid)

    while True:
        model_data = data_handler.get_current_action_data_by_label(current_action_guid_label)
        predicted_action_guid = data_handler.get_original_value_by_label(label_group_name="ActionLabel", label=current_action_guid_label)
        action_name = db_context.get_action_name_by_guid(project_id='0', guid=predicted_action_guid)

        predicted_data = model.predict(model_data)

        # for action in predicted_data:
        #     predicted_action_guid = data_handler.get_original_value_by_label(label_group_name="ActionLabel", label=action)
        #     action_name = db_context.get_action_name_by_guid(project_id='0', guid=predicted_action_guid)
        #     print(f'action name => {action_name}')

        target_action_labels = [a for a in predicted_data if a != current_action_guid_label]
        if len(target_action_labels) == 0:
            break

        target_action_label = random.choice(target_action_labels)
        target_action_guid = data_handler.get_original_value_by_label(label_group_name="ActionLabel", label=target_action_label)

        current_action_name = db_context.get_action_name_by_guid(project_id='0', guid=predicted_action_guid)
        next_action_name = db_context.get_action_name_by_guid(project_id='0', guid=target_action_guid)

        print(f'Moving : {current_action_name} => {next_action_name}')

        if current_action_guid_label == target_action_label:
            break

        current_action_guid_label = target_action_label



    graphy = ClassificationGraphy.ClassificationGraphy(data=_data[['ActionId', 'Next']],
                                                       x_label="Current Action",
                                                       y_label="Next Action")
    class_graphy = graphy
    action_id = data_handler.get_start_entry()
    neighbour_nodes = data_handler.get_current_neighbor_nodes(action_id)

    while True:
        next_action_id, next_action_neighbour_nodes = _data_preprocessing.get_next_action(action_id)

        if len(next_action_neighbour_nodes) == 0:
            print('Auto Scenario generator is suspended')
            break  

        current_action_page = _data_preprocessing.get_page_by_action_id(action_id)
        next_action_page = _data_preprocessing.get_page_by_action_id(action_id)

        class_graphy.Run(action_id, current_action_page, neighbour_nodes)

        # update current action id and neighbour nodes
        action_id = next_action_id
        neighbour_nodes = next_action_neighbour_nodes

    # std = StandardScaler()
    # scaled = std.fit(_data[['Next_Category']].to_numpy())
    # scaled = pd.DataFrame(scaled, columns='Next_Category')

    # _data.drop(columns=['Next_Category'], axis=1, inplace=True)
    # _data = _data.merge(scaled, left_index=True, right_index=True, how = "left")
    # print(_data.head())

    # labelled_dummy_df.to_csv("data_dummy.csv", sep=',')


if __name__ == "__main__":
    run()
