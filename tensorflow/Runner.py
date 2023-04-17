import pandas as pd
import random
from numpy.core.defchararray import strip

from Tools.GraphLibrary import ClassificationGraphy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
data = pd.read_csv("../action_distance.csv")
data.columns = [c.strip() for c in data.columns]

_data = data.loc[:, ["ActionId", "Next", "distance", "rating"]]


# collect unique action Ids
unique_action_id_collections = []
action_id_collections = list(_data['ActionId'].values) + list(_data['Next'].values)
for a in action_id_collections:
    if a not in unique_action_id_collections:
        unique_action_id_collections.append(a)


class_graphy = ClassificationGraphy.ClassificationGraphy(_data[['ActionId', 'Next']], "Current Action",
                                                                 "Next Action")

# Feature label
le = LabelEncoder()
df_category = pd.DataFrame(pd.Series(unique_action_id_collections), columns=['Action_Id_Category'])
df_category['Action_Id_Category_label'] = le.fit_transform(df_category['Action_Id_Category'])

for x, y in zip(df_category['Action_Id_Category'].values, df_category['Action_Id_Category_label'].values):
    _data['ActionId'] = _data['ActionId'].replace([x],y)
    _data['Next'] = _data['Next'].replace([x], y)






# Generate dummy columns for the Next
labelled_dummy_df = pd.get_dummies(data=_data, columns=['Next'])
labelled_dummy_df.to_csv("data_dummy.csv", sep=',')
labelled_dummy_df_next_columns = [c for c in labelled_dummy_df.columns if "Next" in c]

for column in labelled_dummy_df_next_columns:
    labelled_dummy_df[column] = labelled_dummy_df[column].astype(int)


start_entry = labelled_dummy_df.iloc[0]['ActionId']
action_id = df_category.loc[df_category['Action_Id_Category_label'] == start_entry]['Action_Id_Category'].iloc[0]




while True:
    # get a labelled id
    _action_id_label = df_category.loc[df_category['Action_Id_Category'] == action_id]['Action_Id_Category_label'].iloc[0]
    _labelled_neighbour_nodes = _data.loc[_data['ActionId'] == _action_id_label]['Next']


    neighbour_nodes = []
    for _n_nodes in _labelled_neighbour_nodes:
        neighbour_nodes.append(df_category.loc[df_category['Action_Id_Category_label'] == _n_nodes].iloc[0]['Action_Id_Category'])


    _next_action_label_id = random.choice(list(neighbour_nodes))

    class_graphy.Run(action_id, neighbour_nodes, _next_action_label_id)

    action_id = _next_action_label_id

# std = StandardScaler()
# scaled = std.fit(_data[['Next_Category']].to_numpy())
# scaled = pd.DataFrame(scaled, columns='Next_Category')


#_data.drop(columns=['Next_Category'], axis=1, inplace=True)
#_data = _data.merge(scaled, left_index=True, right_index=True, how = "left")
#print(_data.head())


labelled_dummy_df.to_csv("data_dummy.csv", sep=',')




