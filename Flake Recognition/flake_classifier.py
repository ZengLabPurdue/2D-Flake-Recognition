import tensorflow as tf
from tkinter import filedialog
import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

def train_flake_classifier():

    folder_path = filedialog.askdirectory()

    if not folder_path:
        print("No folder selected. Exiting.")
        exit()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.1,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.1,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Classes:", class_names)

    def pad_to_square(image, label):
        image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(pad_to_square, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(pad_to_square, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        data_augmentation,

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save("flake_classifier_tf.keras")

    print("Model saved.")

if __name__ == "__main__":
    train_flake_classifier()