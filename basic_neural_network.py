
# Import necessary libraries
# TensorFlow is used for building and training the neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential  # For creating a linear stack of layers
from tensorflow.keras.layers import Dense      # For adding fully connected layers
from sklearn.model_selection import train_test_split  # To split the dataset
from sklearn.datasets import make_classification      # To generate a toy dataset

# Generate a toy dataset
# make_classification creates a synthetic classification dataset with a specific number of features, classes, and samples.
X, y = make_classification(n_samples=1000,      # Number of samples
                           n_features=20,       # Number of features
                           n_informative=15,    # Number of informative features
                           n_redundant=5,       # Number of redundant features
                           n_classes=2,         # Number of target classes (binary classification)
                           random_state=42)     # Ensures reproducibility

# Split the data into training and testing sets
# train_test_split divides the dataset into training and testing portions.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
# Sequential allows layers to be stacked one after another.
model = Sequential()

# Input layer (first hidden layer)
# Dense creates a fully connected layer with 64 neurons and ReLU activation.
# input_dim specifies the number of input features (20 in this case).
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))

# Add another hidden layer
# This layer also has 32 neurons and uses the ReLU activation function.
model.add(Dense(32, activation='relu'))

# Output layer
# Since this is a binary classification problem, we use a single neuron with a sigmoid activation.
# Sigmoid outputs values between 0 and 1, suitable for binary classification.
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# Compiling configures the model for training.
# Loss function: Binary crossentropy, suitable for binary classification tasks.
# Optimizer: Adam, an adaptive learning rate optimizer.
# Metrics: Accuracy, to track the proportion of correctly predicted instances.
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
# model.fit trains the model for a fixed number of epochs.
history = model.fit(X_train,               # Training input data
                    y_train,               # Training target data
                    epochs=20,             # Number of times the model sees the entire dataset
                    batch_size=32,         # Number of samples per gradient update
                    validation_split=0.2)  # Portion of training data used for validation

# Evaluate the model
# model.evaluate computes the loss and accuracy on the test set.
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.2f}")

# Plot training and validation accuracy
import matplotlib.pyplot as plt

# Extract accuracy and validation accuracy from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Extract number of epochs
epochs = range(1, len(train_acc) + 1)

# Plot the accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
