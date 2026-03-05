def ordinal_fit(data):
    """Create mapping dictionary for ordinal encoding."""
    mapping = {}
    index = 0
    for value in data:
        if value not in mapping:
            mapping[value] = index
            index += 1
    return mapping

def ordinal_transform(data, mapping):
    """Transform categorical data using mapping."""
    encoded = []
    for value in data:
        if value in mapping:
            encoded.append(mapping[value])
    return encoded

def ordinal_fit_transform(data):
    mapping = ordinal_fit(data)
    encoded = ordinal_transform(data, mapping)
    return encoded, mapping

"""--------------onehot encoding--------------"""

def onehot_fit(data):
    """Get unique categories."""
    categories = []
    for value in data:
        if value not in categories:
            categories.append(value)
    return categories

def onehot_transform(data, categories):
    """Convert categorical values into one-hot vectors."""
    encoded = []
    for value in data:
        if value not in categories:
            raise ValueError(f"Unknown category: {value}")
        vector = [0] * len(categories)
        index = categories.index(value)
        vector[index] = 1
        encoded.append(vector)
    return encoded

def onehot_fit_transform(data):
    categories = onehot_fit(data)
    encoded = onehot_transform(data, categories)
    return encoded, categories