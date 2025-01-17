import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
#import keras
import wandb
wandb.login()

# Initialize MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Adjust the batch size by the number of GPUs
num_gpus = strategy.num_replicas_in_sync
scaled_batch_size = int(128) # * num_gpus)

# Configuration dictionary
config = dict(
    batch_size=scaled_batch_size,
    model_name='U-Net Simple',
    epochs=100,
    init_learning_rate=0.0001,
    lr_decay_rate=0.1,
    optimizer='adam',
    loss_fn='mean_squared_error',
    earlystopping_patience=10,
    metrics=[tf.keras.metrics.KLDivergence(), tf.keras.metrics.MeanAbsoluteError(),
             tf.keras.metrics.MeanAbsolutePercentageError()]
)

from wandb.integration.keras import WandbCallback

wandb.init(project='SiSAD', config=config) #, id='e8ic31n9', resume = 'must')


def _parse_function(proto):
    keys_to_features = {
        'images': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    images = tf.io.parse_tensor(parsed_features['images'], out_type=tf.float32)
    images = tf.reshape(images, [3, 63, 63])  # Three images each 63x63
    # Replace any NaNs with zeros in the parsed images
    images = tf.where(tf.math.is_nan(images), tf.zeros_like(images), images)
    return images

@tf.function
def normalize_images(images):
    img1, img2, diff_img = tf.unstack(images, axis=0)  # Unstack into three separate images
    combined = tf.stack([img1, img2], axis=0)  # Stack img1 and img2 for min/max calculation
    min_val = tf.reduce_min(combined)
    max_val = tf.reduce_max(combined)
    img1_rescaled = (img1 - min_val) / (max_val - min_val)
    img2_rescaled = (img2 - min_val) / (max_val - min_val)
    return img1_rescaled, img2_rescaled, diff_img

@tf.function
def normalize_difference_based_on_range(img1, img2, diff_img):
    difference = img1 - img2
    min_diff = tf.reduce_min(difference, keepdims=True)
    max_diff = tf.reduce_max(difference, keepdims=True)
    diff_img_rescaled = 2 * (diff_img - min_diff) / (max_diff - min_diff) - 1
    return diff_img_rescaled

@tf.function
def preprocess_image(images):
    img1, img2, diff_img = normalize_images(images)
    diff_img_rescaled = normalize_difference_based_on_range(images[0], images[1], images[2])
    input_img = tf.stack([img1, img2], axis=-1)  # Stack along the channel dimension
    input_img = tf.image.crop_to_bounding_box(input_img, 7, 7, 48, 48)  # Crop

    diff_img_rescaled = tf.expand_dims(diff_img_rescaled, axis=-1)  # Ensure correct shape for output
    diff_img_rescaled = tf.image.crop_to_bounding_box(diff_img_rescaled, 7, 7, 48, 48)  # Crop

    return input_img, diff_img_rescaled

def load_dataset(tfrecord_paths, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Repeat the dataset indefinitely
    return dataset

# Setup and test
tfrecord_folder_path = "/home/nicosepulveda/astro/bridge/TFRecords"
all_tfrecord_paths = [str(path) for path in Path(tfrecord_folder_path).glob('*.tfrecord')]#[0:100000]
train_paths, test_val_paths = train_test_split(all_tfrecord_paths, test_size=0.3, random_state=1)
val_paths, test_paths = train_test_split(test_val_paths, test_size=0.5, random_state=2)

print(f"Number of sources in training set: {len(train_paths)*2048}")
print(f"Number of sources in validation set: {len(val_paths)*2048}")
print(f"Number of sources in test set: {len(test_paths)*2048}")

# Use strategy to distribute the datasets
train_dataset = load_dataset(train_paths, batch_size=wandb.config['batch_size'])
val_dataset = load_dataset(val_paths, batch_size=wandb.config['batch_size'])

# Distribute the datasets across multiple GPUs
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Contraction Path
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Activation('relu')(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Repeat for additional contraction blocks with increasing filters
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Activation('relu')(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Activation('relu')(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Activation('relu')(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Fully Convolutional Bottleneck
    bn = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    bn = Activation('relu')(bn)
    bn = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn)
    bn = Activation('relu', name='pre_bottleneck')(bn)

    # Global Average Pooling for clustering (optional, remove if not clustering directly from this layer)
    # gap = GlobalAveragePooling2D(name='bottleneck')(bn)

    # Expansive Path
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = Activation('relu')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])  # Skip connection from c3
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])  # Skip connection from c2
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Activation('relu')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = Activation('relu')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])  # Skip connection from c1
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Activation('relu')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1), activation='linear')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name="Custom_U-Net")

    return model

# Adjust the input shape for your 63x63 images with 2 channels (concatenated images)
input_shape = (48, 48, 2)  # Adjusted for the actual size of your images

# Use the strategy scope to distribute the model
with strategy.scope():
    model = build_unet(input_shape)
    model.summary()

    #################################
    # Compile Model
    #################################

    # Callbacks remain unchanged except for Wandb callbacks, which are removed

    # Compile and Train Model
    opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

    # Assuming `model` is your model instance
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.KLDivergence(), tf.keras.metrics.MeanAbsoluteError(),
                          tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.MeanSquaredError()])

    # Configure the early stopping and learning rate scheduler callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], restore_best_weights=True
    )

    def lr_scheduler(epoch, lr):
        # log the current learning rate onto W&B
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        wandb.log({'learning_rate': lr}, commit=False)
        if epoch < 7:
            return lr
        else:
            return lr * tf.math.exp(-config['lr_decay_rate'])

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

    # Initialize the W&B run
    # Define WandbCallback for experiment tracking
    wandb_callback = WandbCallback(
                              monitor='val_loss',
                              log_weights=True,
                              log_evaluation=True)

    # callbacks
    callbacks = [early_stop, wandb_callback, lr_callback]

    # Number of training examples
    num_train_samples = len(train_paths) * 2048  # Adjust 2048 based on your actual number of samples per TFRecord or calculation logic
    num_val_samples = len(val_paths) * 2048  # Same adjustment as above

    train_steps_per_epoch = np.ceil(num_train_samples / config['batch_size']).astype(int)
    val_steps_per_epoch = np.ceil(num_val_samples / config['batch_size']).astype(int)

    # Train the model
    history = model.fit(train_dist_dataset,
                        epochs=40,  # Adjust based on the complexity of the model and the dataset size
                        # Originalmente 100 epoch pero para 10000 tfrecords
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_dist_dataset,
                        validation_steps=val_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=2
                        )

wandb.finish()
