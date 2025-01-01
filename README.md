The Titanic dataset is loaded from a public URL. This dataset contains various features related to passengers (e.g., class, sex, age, fare) and whether they survived or not.

data = pd.read_csv(url)
The dataset is read into a pandas DataFrame.

data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]
The dataset is filtered to keep only the relevant columns: Pclass, Sex, Age, Fare, and Survived.

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
The Sex column, which contains categorical values ('male' and 'female'), is mapped to numeric values (0 for male and 1 for female).

data = data.dropna()
Any rows with missing values are dropped from the dataset.
X = data[['Pclass', 'Sex', 'Age', 'Fare']].values
The feature matrix X is created by extracting the columns Pclass, Sex, Age, and Fare.

y = data['Survived'].values
The target vector y is created by extracting the Survived column, which indicates whether the passenger survived (1) or not (0).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
The dataset is split into training and testing sets using an 80-20 split. X_train and y_train are for training, while X_test and y_test are for testing.
scaler = StandardScaler()
A StandardScaler is used to scale the features so that they have zero mean and unit variance.

X_train = scaler.fit_transform(X_train)
The training data is scaled.

X_test = scaler.transform(X_test)
The test data is scaled using the same scaler that was fitted to the training data.
class PerceptronWithVisualization
A custom perceptron class is defined with methods for training (fit) and predicting (predict). The perceptron uses a simple linear model to predict the output based on the weighted sum of the inputs.

fit method:
This method trains the perceptron by iterating through the data for a specified number of iterations (n_iterations). It applies the perceptron learning rule to update the weights and bias based on the error between the predicted and actual values.

predict method:
This method makes predictions by calculating the weighted sum of the inputs and applying a step activation function to classify the data as 1 or 0 (survived or not).
model = PerceptronWithVisualization(learning_rate=0.01, n_iterations=1000)
A perceptron model is created with a learning rate of 0.01 and 1000 iterations.

model.fit(X_train, y_train)
The perceptron is trained on the training data.
plt.plot(range(1, 1001), model.errors, color='b')
This plot shows the number of errors in the predictions over the training iterations, which demonstrates how the perceptron improves over time.
y_pred_custom = model.predict(X_test)
The trained custom perceptron model is used to make predictions on the test set.

accuracy_custom = np.mean(y_pred_custom == y_test)
The accuracy of the custom perceptron is calculated as the percentage of correct predictions on the test set.

sklearn_model = Perceptron(max_iter=1000, eta0=0.01)
A Scikit-learn perceptron model is created with the same parameters (1000 iterations and learning rate of 0.01).

sklearn_model.fit(X_train, y_train)
The Scikit-learn perceptron is trained on the training data.

y_pred_sklearn = sklearn_model.predict(X_test)
The trained Scikit-learn model is used to make predictions on the test set.

accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
The accuracy of the Scikit-learn perceptron is calculated.
plt.bar(['Custom Perceptron', 'Scikit-learn Perceptron'], [accuracy_custom, accuracy_sklearn])
A bar plot is created to compare the accuracy of the custom perceptron and the Scikit-learn perceptron.
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap='bwr', alpha=0.7)
A scatter plot is created to visualize the distribution of the Age and Fare features in the training data, with different colors indicating survival (0 or 1).

plt.colorbar(label='Survived')
A color bar is added to indicate the survival status of the passengers.
