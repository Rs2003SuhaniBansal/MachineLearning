import math
def entropy(labels):
    total = len(labels)
    classes = {}
    for label in labels:
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1

    entropy = 0.00
    for i in classes.values():
        P = i/total
        entropy = entropy - P * math.log2(P)

    return entropy

"""This function calculates the weighted entropy and the Information Gain"""
def information_gain(parent, children):
    n = len(parent)
    H_parent = entropy(parent)
    H_weighted = 0
    for child in children:
        H_weighted += len(child)/n * entropy(child)
    IG = H_parent - H_weighted
    return IG

parent = ["No","No","Yes","Yes","No","Yes","Yes","No"]

sunny = ["No","No","No"]
overcast = ["Yes","Yes"]
rainy = ["Yes","No","Yes"]

children = [sunny, overcast, rainy]

ig = information_gain(parent, children)

print("Information Gain:", ig)