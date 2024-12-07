import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense    
from sklearn.model_selection import train_test_split  
from sklearn.datasets import make_classification    

X, y = make_classification(n_samples=1000,      
                           n_features=20,       # Number of features
                           n_informative=15,    # Number of informative features
                           n_redundant=5,       # Number of redundant features
                           n_classes=2,         # Number of target classes (binary classification)
                           random_state=42)     # Ensures reproducibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X_train,               
                    y_train,               
                    epochs=20,           
                    batch_size=32,     
                    validation_split=0.2) 


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.2f}")

import matplotlib.pyplot as plt

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_acc) + 1)


plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
