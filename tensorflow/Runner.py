import pandas as pd
from numpy.core.defchararray import strip

from Tools.GraphLibrary import ClassificationGraphy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
data = pd.read_csv("../action_distance.csv")
data.columns = [c.strip() for c in data.columns]

_data = data.loc[:, ["ActionId", "Next", "distance", "rating"]]

#_data['Next'] = _data['Next'].astype(int)
df = pd.get_dummies(data=_data, columns=['Next'])
df_next = df.loc[:, [4]]



'''
df_boolean_data = df.loc[:, ~df.columns.isin(["ActionId", "Next", "distance", "rating"])]
le = LabelEncoder()
_data['Next_Category'] = le.fit_transform(_data['Next'])

print(_data.head())

std = StandardScaler()
scaled = std.fit(_data[['Next_Category']].to_numpy())
scaled = pd.DataFrame(scaled, columns='Next_Category')
'''

#_data.drop(columns=['Next_Category'], axis=1, inplace=True)
#_data = _data.merge(scaled, left_index=True, right_index=True, how = "left")
#print(_data.head())


df.to_csv("data_dummy.csv", sep=',')

class_graphy = ClassificationGraphy.ClassificationGraphy(_data, "Current Action", "Next Action")
class_graphy.Run()


