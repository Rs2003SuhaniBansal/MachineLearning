import math
"""the function entropy will calculate the entropy of individual conditions
in the parent data (for example sunny, rainy, overcast) and the parent data itself."""
def entropy(data, labels, condition=None):

    w = []
    """if condition is not specified (for parent) the list is equal to the labels."""
    if condition is None:
        w = labels
    else:
        """if condition is specified it is appended to the list
        such that the list will contain only the label of the specified condition. """
        for i in range(len(data)):
            if data[i] == condition:
                w.append(labels[i])

    total = len(w)

    """the classes dict will contain the count of the labels in the above list."""
    classes = {}
    for label in w:
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1
    """here the entropy is calculated for each class."""
    entropy = 0.00
    for i in classes.values():
        P = i/total
        entropy = entropy - P * math.log2(P)
    return entropy

weather = ["Sunny","Sunny","Overcast","Rainy","Rainy","Rainy","Overcast","Sunny"]
play = ["No","No","Yes","Yes","No","Yes","Yes","No"]

H_root = entropy(weather, play)
H_sunny = entropy(weather, play, condition="Sunny")
H_overcast = entropy(weather, play, condition="Overcast")
H_rainy = entropy(weather, play, condition="Rainy")

print(f" H_root: {H_root}, "
      f"\nH_sunny: {H_sunny}, "
      f"\nH_overcast: \n{H_overcast}, "
      f"\nH_rainy: {H_rainy}")