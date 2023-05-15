from Tools.GraphLibrary import ClassificationGraphy
from numpy.core.defchararray import strip
from data_preprocessing import data_preprocessing
from knn_model import KNN_model

def run():
    preprocessing = data_preprocessing()
    model = KNN_model()
    _data_preprocessing = preprocessing
    _data_preprocessing.init()
    _data = _data_preprocessing.get_src_data()
    knn_seed_data = _data_preprocessing.get_compiled_data()

    model.get_seed_data(knn_seed_data)
    model.Run()



    graphy = ClassificationGraphy.ClassificationGraphy(data=_data[['ActionId', 'Next']],
                                                       x_label="Current Action",
                                                       y_label="Next Action")
    class_graphy = graphy
    action_id = _data_preprocessing.get_start_entry()
    neighbour_nodes = _data_preprocessing.get_current_neighbor_nodes(action_id)

    while True:
        next_action_id, next_action_neighbour_nodes = _data_preprocessing.get_next_action(action_id)
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
