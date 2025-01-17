import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import wandb
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, concatenate, MaxPooling2D, Dropout, Dense


wandb.login()

# Initialize MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Adjust the batch size by the number of GPUs
num_gpus = strategy.num_replicas_in_sync
scaled_batch_size = int(2048)  # * num_gpus

# Configuration dictionary
config = dict(
    batch_size=scaled_batch_size,
    model_name='Attention Unet',
    epochs=100,
    init_learning_rate=0.0001,
    lr_decay_rate=0.1,
    optimizer='adam',
    loss_fn='mean_squared_error',
    earlystopping_patience=10,
    metrics=[keras.metrics.KLDivergence(), keras.metrics.MeanAbsoluteError(),
             keras.metrics.MeanAbsolutePercentageError()]
)

from wandb.keras import WandbCallback

wandb.init(project='SiSAD', config=config, id='y3gyuo3z', resume = 'allow')

def _parse_function(proto):
    keys_to_features = {
        'images': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    images = tf.io.parse_tensor(parsed_features['images'], out_type=tf.float32)
    images = tf.reshape(images, [3, 63, 63])  # Three images each 63x63
    images = tf.where(tf.math.is_nan(images), tf.zeros_like(images), images)
    return images

@tf.function
def normalize_images(images):
    img1, img2, diff_img = tf.unstack(images, axis=0)
    combined = tf.stack([img1, img2], axis=0)
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
    input_img = tf.stack([img1, img2], axis=-1)
    input_img = tf.image.crop_to_bounding_box(input_img, 7, 7, 48, 48)

    diff_img_rescaled = tf.expand_dims(diff_img_rescaled, axis=-1)
    diff_img_rescaled = tf.image.crop_to_bounding_box(diff_img_rescaled, 7, 7, 48, 48)

    return input_img, diff_img_rescaled

def load_dataset(tfrecord_paths, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

# Setup and test
tfrecord_folder_path = "/home/nicosepulveda/astro/bridge/TFRecords"
all_tfrecord_paths = [str(path) for path in Path(tfrecord_folder_path).glob('*.tfrecord')][0:10000]
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

def conv_block(x, num_filters, dropout_rate=0.0, name=None):
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    if name:
        x = Activation("relu", name=name)(x)
    return x

def attention_gate(g, s, num_filters, dropout_rate=0.0):
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)
    Ws = Conv2D(num_filters, 1, padding="same")(s)
    Ws = BatchNormalization()(Ws)
    out = Activation("relu")(Wg + Ws)
    out = Conv2D(1, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
    if dropout_rate > 0.0:
        out = Dropout(dropout_rate)(out)
    return out * s

def encoder_block(x, num_filters, dropout_rate=0.0):
    x = conv_block(x, num_filters, dropout_rate=dropout_rate)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(x, s, num_filters, dropout_rate=0.0):
    x = UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters, dropout_rate=dropout_rate)
    x = concatenate([x, s])
    x = conv_block(x, num_filters, dropout_rate=dropout_rate)
    return x

def attention_unet(input_shape=(48, 48, 2), dropout_rate=0.1):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 48, dropout_rate)
    s2, p2 = encoder_block(p1, 96, dropout_rate)
    s3, p3 = encoder_block(p2, 192, dropout_rate)
    b1 = conv_block(p3, 512, dropout_rate, name='bottleneck')
    d1 = decoder_block(b1, s3, 192, dropout_rate)
    d2 = decoder_block(d1, s2, 96, dropout_rate)
    d3 = decoder_block(d2, s1, 48, dropout_rate)
    outputs = Conv2D(1, 1, padding="same", activation="linear")(d3)
    model = Model(inputs, outputs, name="Attention-UNET")
    return model

# Adjust the input shape for your 48x48 images with 2 channels (concatenated images)
input_shape = (48, 48, 2)  # Adjusted for the actual size of your images

with strategy.scope():
    model = attention_unet(input_shape)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=[keras.metrics.KLDivergence(), keras.metrics.MeanAbsoluteError(),
                           keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanSquaredError()])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], restore_best_weights=True
    )

    def lr_scheduler(epoch, lr):
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        wandb.log({'learning_rate': lr}, commit=False)
        if epoch < 7:
            return lr
        else:
            return lr * tf.math.exp(-config['lr_decay_rate'])

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    wandb_callback = WandbCallback(
        monitor='val_loss',
        log_weights=True,
        log_evaluation=True
    )

    callbacks = [early_stop, wandb_callback, lr_callback]

    num_train_samples = len(train_paths) * 2048
    num_val_samples = len(val_paths) * 2048

    train_steps_per_epoch = np.ceil(num_train_samples / config['batch_size']).astype(int)
    val_steps_per_epoch = np.ceil(num_val_samples / config['batch_size']).astype(int)

    history = model.fit(train_dist_dataset,
                        epochs=config['epochs'],
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_dist_dataset,
                        validation_steps=val_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=2)

wandb.finish()
