def encode_value(value, mapping):
    return mapping.get(value.lower(), 0)

# These mappings should match the encoding in model_training.py.
category_map = {'tech': 0, 'finance': 1, 'health': 2, 'education': 3, 'ecommerce': 4}
state_map = {'ca': 0, 'ny': 1, 'tx': 2, 'wa': 3}
city_map = {'san francisco': 0, 'new york': 1, 'austin': 2, 'seattle': 3}
