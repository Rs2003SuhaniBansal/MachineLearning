
def load_data(datafile):
    file = open(datafile, "r")
    content = file.read()

def div_sets(content):
    x = content.drop(columns=["disease_score_fluct"])
    y = content["disease_score_fluct"]
