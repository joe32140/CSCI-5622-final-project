# CSCI-5622-final-project

## Feature Extraction
Download `extracted_data.pkl` [Here](https://drive.google.com/file/d/1X5AQI4yvl3KyhVZMQvdbSoGv6QHxHCzC/view?usp=sharing).

Date structure:  
```
data = {  
    "train":[(recipe_id, list_of_image_features, list_of_ingredients)],  
    "val":[...],  
    "test":[...]  
}
```

```python
import pickle
data = pickle.load(open('extracted_data.pkl', 'rb'))
```

## Ingredients to Cuisine Model

- The notebook is available in whats-cooking/ directory. SVM gives the best accuracy on the test set. I have to fine tune it a bit more probably tweak a few hyperparameters.

- We should use this with the 1M+ dataset to build the cuisine for each entry in the training set.

- The model implementation is really simple. We are doing a TF(Term frequency)-IDF(Inverse document frequency) representation for the ingredients corpus and then training it. Maybe, we could try a BOW representation as well and see how it performs.

- To run the model, download the dataset from kaggle and place them in the ```whats-cooking``` directory. There would be two files - name them train.json and test.json .
