# music-genre-ai
This is a simple machine learning code that gets a dataframe from a pseudo collection of data from a csv, it is all fake data and extremelly simple for educational purposes, as I am a beginner in AI. The code below gets an input (age and gender) and an ouput (genra), then it creates a model, trains the model with an DecisionTreeClassifier algorithm and then I create the predictions with the test inputs, which corresponds to 20% of the total data. Overall, the accuracy is of 75% to 100%.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# joblib.dump(model, 'music-recommender.joblib')

model = joblib.load('music-recommender.joblib')
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'], class_names=sorted(y.unique()),
                    label='all', rounded=True, filled=True)

```

## Data prediction visualization
<p align="center">
  <img src="https://raw.githubusercontent.com/nandowastaken/music-genre-ai/main/predictionsVisualization.png">
</p>
