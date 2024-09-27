```Python
#Importing Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Importing Data
df = pd.read_csv('nfl_combine.csv')
data = df
```

```Python
#Changing Categorical Variables to Numeric
def to_numeric(col, categories):
    return categories.index(col)

columns = ['Pos', 'School', 'Drafted']

for column in columns:
    categories = data[column].unique().tolist()
    data[column] = data[column].apply(lambda x: to_numeric(x, categories))
```

```Python
#Converting Height to Inches
def convert_height(height_str):
    if pd.notna(height_str):
        feet, inches = map(int, height_str.split('-'))
        total_inches = (feet * 12) + inches
        return total_inches
    else:
        return 0
    
data['Height'] = data['Height'].apply(convert_height)

#Filling NaN with 0s and Forcing all Data to Numeric
data.fillna(0, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
```

```Python
#Setting Test Variables
variables = ['Pos',
             'School',
             'Height',
             'Weight',
             '40yd',
             'Vertical',
             'Bench',
             'Broad Jump',
             '3Cone',
             'Shuttle']

#Describing Data
data = data[variables]
data.describe()
```

```Python
#Plotting Draft Counts
plt.figure(figsize=(8,6))
sns.countplot(df, x='Drafted')
plt.title('Number of Athletes Drafted')
plt.ylabel('Number Drafted')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()
```

```Python
#Setting Predictors and Predictees
X = data[variables].values
y = df['Drafted'].values

#Splting Data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.2)

#Creating Logistic Regression Model
model = LogisticRegression(max_iter=1000)

#Fiting the Model
model.fit(X_train, y_train)

#Running the Model
predict = model.predict(X_test)

#Printing Model Accuracy as a Percent
accuracy = metrics.accuracy_score(y_test, predict)
print(f'Accuracy: {accuracy * 100}%')
```

```Python
#Pulling Variables of Most Importance
coeff = abs(model.coef_[0])
variable_importance_df = pd.DataFrame({'Variable': variables, 'Importance': coeff})

#Plotting Variable Importance
plt.figure(figsize=(10,6))
sns.barplot(variable_importance_df, x='Variable', y='Importance', hue=sns.color_palette(), legend=False)
plt.title('Most Important Varibales for Getting Drafted')
plt.xticks(rotation=-30)
```