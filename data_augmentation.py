import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def apply_data_augmentation(train_data_generator):
    # Define the data augmentation parameters
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Apply data augmentation to the training data
    return data_augmentation.flow_from_directory(
        directory="path_to_training_data_directory",
        target_size=(224, 224),  # Adjust the target size as needed
        batch_size=32,  # Adjust the batch size as needed
        class_mode="binary"
    )


# Example usage
if __name__ == "__main__":
    train_data_generator = apply_data_augmentation(train_data_generator)
    # You can use train_data_generator in your main script for training
