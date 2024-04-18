from sklearn.model_selection import cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
from scikeras.wrappers import KerasClassifier

RELU='relu'
SIGMOID='sigmoid'

# завантажуємо датасет
dataset = read_csv("../sonar.csv", header=None).values
# розділяємо в інпут (X) та аутпут (Y) змінні
input_x = dataset[:,0:60].astype(float)
input_y = dataset[:,60]

# енкодемо занчення в інтеджери
encoder = LabelEncoder()
encoder.fit(input_y)
encoded_Y = encoder.transform(input_y)

name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Rock Mine label mapping:", name_mapping)
 
# естімейшини 
estmtrs = []
estmtrs.append(('standardize', StandardScaler()))
def create_baseline():
    # створюємо модель і задаємо леєри
    mdl = Sequential()
    mdl.add(Dense(60, input_shape=(60,), activation=RELU))
    mdl.add(Dense(30,  activation=RELU))
    mdl.add(Dense(1, activation=SIGMOID))
    # компілюємо модель
    sgd = SGD(learning_rate=0.01, momentum=0.8)
    mdl.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return mdl
estmtrs.append(('mlp', KerasClassifier(model=create_baseline, epochs=300, batch_size=16, verbose=0)))
# пайплайн
ppln = Pipeline(estmtrs)
kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_validate(ppln, input_x, encoded_Y, cv=kfold, return_train_score=True)

# Print train and test accuracies
print("Train accuracy: %.2f%% " % (results['train_score'].mean()*100))
print("Test accuracy: %.2f%% " % (results['test_score'].mean()*100))