# CSCI-5622-final-project

## Feature Extraction
Date structure:
data = {
    "train":[(recipe_id, list_of_image_features, list_of_ingredients)],
    "val":[...],
    "test":[...]
}


```python
import pickle
data = pickle.load(open('extracted_data.pkl', 'rb'))
```