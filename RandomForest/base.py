from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import ensemble
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

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

# input_x = SelectKBest(f_classif, k=50).fit_transform(input_x, encoded_Y)

# створюємо модель
model = ensemble.RandomForestClassifier(n_estimators=55, criterion="gini", max_depth=20, max_features="log2", min_samples_split=2, max_leaf_nodes=30, bootstrap=False)

# Тренуємо та оцінюємо точність
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, input_x, encoded_Y, cv=kfold)
print("Модель: %.2f%% " % (results.mean()*100))
