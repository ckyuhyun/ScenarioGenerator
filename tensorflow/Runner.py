import pandas as pd
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

# Feature label
le = LabelEncoder()
df_label = pd.DataFrame(pd.Series(unique_action_id_collections), columns=['Action_Id_Category'])
df_label['Action_Id_Category_label'] = le.fit_transform(df_label['Action_Id_Category'])

for x, y in zip(df_label['Action_Id_Category'].values, df_label['Action_Id_Category_label'].values):
    _data['ActionId'] = _data['ActionId'].replace([x],y)
    _data['Next'] = _data['Next'].replace([x], y)

# turn boolean value into numerical value on Next columns

# Generate dummy columns for the Next
df = pd.get_dummies(data=_data, columns=['Next'])
df_next_columns = [c for c in df.columns if "Next" in c]

for column in df_next_columns:
    df[column] = df[column].astype(int)




# std = StandardScaler()
# scaled = std.fit(_data[['Next_Category']].to_numpy())
# scaled = pd.DataFrame(scaled, columns='Next_Category')


#_data.drop(columns=['Next_Category'], axis=1, inplace=True)
#_data = _data.merge(scaled, left_index=True, right_index=True, how = "left")
#print(_data.head())


df.to_csv("data_dummy.csv", sep=',')

class_graphy = ClassificationGraphy.ClassificationGraphy(_data, "Current Action", "Next Action")
class_graphy.Run()


