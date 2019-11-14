# CSCI-5622-final-project

## Feature Extraction
Download `extracted_data.pkl` [Here](https://drive.google.com/file/d/1X5AQI4yvl3KyhVZMQvdbSoGv6QHxHCzC/view?usp=sharing).

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
