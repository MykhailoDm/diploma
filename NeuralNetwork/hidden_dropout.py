from sklearn.model_selection import cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
from scikeras.wrappers import KerasClassifier

# завантажуємо датасет
dataset = read_csv("../sonar.csv", header=None).values
# розділяємо в інпут (X) та аутпут (Y) змінні
input_x = dataset[:,0:60].astype(float)
input_y = dataset[:,60]
# енкодемо занчення в інтеджери
encoder = LabelEncoder()
encoder.fit(input_y)
encoded_Y = encoder.transform(input_y)
 
# естімейшини 
estmtrs = []
estmtrs.append(('standardize', StandardScaler()))
def pplncreate_with_dropout():
    # створюємо модель і задаємо леєри, Додається Dropout до Hidden Layer
    hidden_dropout_weight = 0.5
    mdl = Sequential()
    mdl.add(Dense(60, input_shape=(60,), activation='relu', kernel_constraint=MaxNorm(3)))
    mdl.add(Dropout(hidden_dropout_weight))
    mdl.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    mdl.add(Dropout(hidden_dropout_weight))
    mdl.add(Dense(1, activation='sigmoid'))
    # компілюємо модель
    sgd = SGD(learning_rate=0.01, momentum=0.8)
    mdl.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return mdl
estmtrs.append(('mlp', KerasClassifier(model=pplncreate_with_dropout, epochs=300, batch_size=16, verbose=0)))
# пайплайн
ppln = Pipeline(estmtrs)
kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_validate(ppln, input_x, encoded_Y, cv=kfold, return_train_score=True)

# Print train and test accuracies
print("Train accuracy: %.2f%% " % (results['train_score'].mean()*100))
print("Test accuracy: %.2f%% " % (results['test_score'].mean()*100))


# The time for scoring the estimator on the test set for each cv split (seconds)
print("Time:", (results['score_time'].mean()))