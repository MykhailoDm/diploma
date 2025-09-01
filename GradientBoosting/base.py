from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

# завантажуємо датасет
dataset = read_csv("../sonar.csv", header=None).values
# розділяємо в data (X) та target (Y) змінні
input_x = dataset[:,0:60].astype(float)
input_y = dataset[:,60]

# енкодемо занчення Y в інтеджери
encoder = LabelEncoder()
encoder.fit(input_y)
encoded_Y = encoder.transform(input_y)
name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Rock Mine label mapping:", name_mapping)

# input_x = SelectKBest(f_classif, k=50).fit_transform(input_x, encoded_Y)

# створюємо модель
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

# Тренуємо та оцінюємо точність
kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_validate(model, input_x, encoded_Y, cv=kfold, return_train_score=True)

# Print train and test accuracies
print("Train accuracy: %.2f%% " % (results['train_score'].mean()*100))
print("Test accuracy: %.2f%% " % (results['test_score'].mean()*100))


# The time for scoring the estimator on the test set for each cv split (seconds)
print("Time:", (results['score_time'].mean()))

param_groups = [
    {
        "n_estimators": 100,
        "learning_rate": 1.0
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.5
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.5
    },
    {
        "n_estimators": 150,
        "learning_rate": 1.0
    },
]

n_estimators = []
n_est = 10
n_est_step = 10
learning_rate = []
lear_rate = 0.1
lear_rate_step = 0.1
acc = []
time = []

for i in range(1, 5): 
    for j in range(1, 5):        
        # створюємо модель
        model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lear_rate, max_depth=1, random_state=42)

        # Тренуємо та оцінюємо точність
        kfold = StratifiedKFold(n_splits=10, shuffle=True)

        results = cross_validate(model, input_x, encoded_Y, cv=kfold, return_train_score=True)

        # Print train and test accuracies
        print("Train accuracy: %.2f%% " % (results['train_score'].mean()*100))
        print("Test accuracy: %.2f%% " % (results['test_score'].mean()*100))


        # The time for scoring the estimator on the test set for each cv split (seconds)
        print("Time:", (results['score_time'].mean()))
        print()

        n_estimators.append(n_est)
        n_est += n_est_step
        learning_rate.append(lear_rate)
        lear_rate += lear_rate_step
        acc.append(results['test_score'].mean()*100)
        time.append(results['score_time'].mean())
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(n_estimators, learning_rate, acc, 'green')
ax.set_xlabel('N Estimators')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Accuracy')
ax.set_title('Gradient Boosting Accuracy')
plt.show()
ax = plt.axes(projection='3d')
ax.plot3D(n_estimators, learning_rate, acc, 'green')
ax.set_xlabel('N Estimators')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Time')
ax.set_title('Gradient Boosting Time')
plt.show()