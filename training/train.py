import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 35
n_classes = 3

# this file is for training the model to classify potatoes, not tomatoes
# the jupyter notebook is for training the model to classify tomatoes

# cpu only, used since tensorflow mac doesn't support GPU for RandomFlip
# cpus = tf.config.experimental.list_physical_devices("CPU")
# tf.config.set_visible_devices([], "GPU")  # hide the GPU
# tf.config.set_visible_devices(cpus[0], "CPU")  # unhide potentially hidden CPU
# tf.config.get_visible_devices()

# gpu only
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.set_visible_devices([], "CPU")  # hide the CPU
# tf.config.set_visible_devices(gpus[0], "GPU")  # unhide potentially hidden GPU
# tf.config.get_visible_devices()


# both
tf.config.get_visible_devices()

# load

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

# data visualization
class_names = dataset.class_names
print(class_names) #prints names
# print(len(dataset)) #prints number of batches (68)
# plt.figure(figsize=(10, 10))
# for img_batch, lbl_batch in dataset.take(1):
#     for i in range(12):
#         plt.subplot(3, 4, i + 1)
#         plt.imshow(img_batch[i].numpy().astype("uint8"))
#         plt.axis("off")
#         plt.title(class_names[lbl_batch[i]])
#     plt.show()


# 80% training, 10% validation (used after epoch, training process), 10% test (used after model is trained)


def get_dataset_partitions_tf(
    ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000
):
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    if shuffle:
        ds.shuffle(shuffle_size, seed=22)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# caches images for next epoch, prefetch fetches next batch before next iteration


# preprocessing, creating layers

# scale the images down to 0-1
resize_and_rescale = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
    ]
)

# data augmentation (in case images are rotated or high contrast or flipped or some other transformation)
with tf.device("/cpu:0"):  # tf metal mac doesnt support RandomFlip for GPU acceleration
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

# make model

# define architecture

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model = models.Sequential(
    [
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ]
)

model.build(input_shape=input_shape)
# model.summary()


# compile neural network

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# train
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
)

# scores = model.evaluate(test_ds)

# def predict(model, img):
#     img_arr = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#     img_arr = tf.expand_dims(img_arr, 0)

#     predictions = model.predict(img_arr)

#     predicted_class = class_names[np.argmax(predictions[0])]
#     conf = round(100*np.max(predictions[0]), 2)
#     return predicted_class, conf

model_version = len(os.listdir("../model")) + 1
model.save(f"../model/{model_version}")
