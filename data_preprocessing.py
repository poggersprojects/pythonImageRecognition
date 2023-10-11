import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Define data paths
train_dir = 'image-recognition/kagglecatsanddogs_5340/PetImages/'
train_cats_dir = 'image-recognition/kagglecatsanddogs_5340/PetImages/Cat'
train_dogs_dir = 'image-recognition/kagglecatsanddogs_5340/PetImages/Dog'

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values to the range [0, 1]
    rotation_range=40,            # Randomly rotate images
    width_shift_range=0.2,        # Randomly shift images horizontally
    height_shift_range=0.2,       # Randomly shift images vertically
    shear_range=0.2,              # Shear transformations
    zoom_range=0.2,               # Randomly zoom in on images
    horizontal_flip=True,         # Randomly flip images horizontally
    fill_mode='nearest'           # Fill in missing pixels with the nearest value
)

# Create data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Binary classification (cat or dog)
)

# Data preprocessing and augmentation settings for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create a validation data generator
val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Binary classification (cat or dog)
)
