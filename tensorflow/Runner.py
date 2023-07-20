import pandas as pd

from Tools.GraphLibrary import ClassificationGraphy
from numpy.core.defchararray import strip
from data_model import data_model
from knn_model import KNN_model
from helper.db_context_helper import dbContext_helper


def run():
    read_data = pd.read_csv('../extended_action_data.csv', index_col=0)

    #df = dl.get_data_with_label(['TestActionGuid'])
    preprocessing = data_model(read_data)
    model = KNN_model()
    db_context = dbContext_helper()

    #_data_preprocessing = preprocessing
    #_data_preprocessing.init()
    #_data = _data_preprocessing.get_src_data()
    knn_seed_data = preprocessing.get_model_seed_data()
    model.get_seed_data(knn_seed_data)
    model.Run()

    model_data = preprocessing.get_current_action_data_by_label(0)

    predicted_data = model.predict(model_data)

    for action in predicted_data:
        predicted_action_guid = preprocessing.get_original_value_by_label(label_group_name="ActionLabel", label=action)
        action_name = db_context.get_action_name_by_guid(project_id='0', guid=predicted_action_guid)
        print(f'action name => {action_name}')


    target_action_guid = preprocessing.get_original_value_by_label(label_group_name="ActionLabel", label= target_action_label)

    print(f'predicted action guid : {predicted_action_guid}')
    print(f'target action guid : {target_action_guid}')




    graphy = ClassificationGraphy.ClassificationGraphy(data=_data[['ActionId', 'Next']],
                                                       x_label="Current Action",
                                                       y_label="Next Action")
    class_graphy = graphy
    action_id = preprocessing.get_start_entry()
    neighbour_nodes = preprocessing.get_current_neighbor_nodes(action_id)

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
