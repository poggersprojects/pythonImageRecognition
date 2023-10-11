import os
import random

# Define the paths to your dataset directories
base_directory = "image-recognition/kagglecatsanddogs_5340/PetImages"
cats_directory = os.path.join(base_directory, "Cat")
dogs_directory = os.path.join(base_directory, "Dog")

# List the image filenames in each directory
cat_images = [os.path.join(cats_directory, filename) for filename in os.listdir(cats_directory)]
dog_images = [os.path.join(dogs_directory, filename) for filename in os.listdir(dogs_directory)]

# Shuffle the image filenames
random.shuffle(cat_images)
random.shuffle(dog_images)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


# Split the data
def split_data(image_list, train_ratio, val_ratio, test_ratio):
    total_images = len(image_list)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_data = image_list[:train_end]
    val_data = image_list[train_end:val_end]
    test_data = image_list[val_end:]

    return train_data, val_data, test_data


cat_train, cat_val, cat_test = split_data(cat_images, train_ratio, val_ratio, test_ratio)
dog_train, dog_val, dog_test = split_data(dog_images, train_ratio, val_ratio, test_ratio)

# Combine and shuffle the data
train_data = cat_train + dog_train
val_data = cat_val + dog_val
test_data = cat_test + dog_test

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)


# Save the split datasets to separate text files
def save_data_to_file(data, filename):
    with open(filename, "w") as file:
        for item in data:
            file.write(f"{item}\n")


save_data_to_file(train_data, "train_data.txt")
save_data_to_file(val_data, "val_data.txt")
save_data_to_file(test_data, "test_data.txt")
