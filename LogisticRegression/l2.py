from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from pandas import read_csv

# завантажуємо датасет
dataset = read_csv("sonar.csv", header=None).values
# розділяємо в data (X) та target (Y) змінні
input_x = dataset[:,0:60].astype(float)
input_y = dataset[:,60]

# енкодемо занчення Y в інтеджери
encoder = LabelEncoder()
encoder.fit(input_y)
encoded_Y = encoder.transform(input_y)
name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Rock Mine label mapping:", name_mapping)

# створюємо модель
model = linear_model.LogisticRegression(penalty='l2', solver='liblinear', max_iter=100, C=1)

# Тренуємо та оцінюємо точність
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, input_x, encoded_Y, cv=kfold)
print("Модель: %.2f%% " % (results.mean()*100))