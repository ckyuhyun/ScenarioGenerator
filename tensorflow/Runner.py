import pandas as pd
from Tools.GraphLibrary import ClassificationGraphy

data = pd.read_csv("../action_distance.csv")

_data = data.loc[:, ["ActionId", "Next", "distance", "rating"]][:5]


print(_data)
class_graphy = ClassificationGraphy.ClassificationGraphy(_data, "Current Action", "Next Action")
class_graphy.Draw()


