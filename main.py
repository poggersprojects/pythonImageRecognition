import tensorflow as tf
from tensorflow import keras
from keras import layers
from data_preprocessing import train_generator, val_generator
from plotting import plot_training_history
from data_augmentation import apply_data_augmentation

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, hence the sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Apply data augmentation to the training data generator
train_generator = apply_data_augmentation(train_generator)

# Print a summary of the model architecture
model.summary()

# Train the model and collect training history
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Use the plotting function from plotting.py
plot_training_history(history)
