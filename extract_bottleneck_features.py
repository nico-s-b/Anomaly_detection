#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from pathlib import Path
import h5py

scaled_batch_size = int(128) # * num_gpus)

def main():
    config = dict(
    batch_size=scaled_batch_size,
    model_name='U-Net Simple',
    epochs=100,
    init_learning_rate=0.0001,
    lr_decay_rate=0.1,
    optimizer='adam',
    loss_fn='mean_squared_error',
    earlystopping_patience=10,
    )
    
    # Assuming your model is saved in this path
    model_path = '/home/nicosepulveda/astro/bridge/model/local_1st_best_model/model-best.h5'
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Define a bottleneck model to extract features from the bottleneck layer and apply global average pooling
    bottleneck_output = model.get_layer('pre_bottleneck').output
    global_avg_pooling = GlobalAveragePooling2D()(bottleneck_output)
    bottleneck_model = Model(inputs=model.input, outputs=global_avg_pooling)
    
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
        diff_img_rescaled = normalize_difference_based_on_range(img1, img2, diff_img)
        input_img = tf.stack([img1, img2], axis=-1)  # Stack along the channel dimension
        input_img = tf.image.crop_to_bounding_box(input_img, 7, 7, 48, 48)  # crop
    
        diff_img_rescaled = tf.expand_dims(diff_img_rescaled, axis=-1)  # Ensure correct shape for output
        diff_img_rescaled = tf.image.crop_to_bounding_box(diff_img_rescaled, 7, 7, 48, 48)  # crop
    
        return input_img, diff_img_rescaled
    
    def parse_tfrecord(example_proto):
        feature_description = {
            'images': tf.io.FixedLenFeature([], tf.string),
            'objectIds': tf.io.FixedLenFeature([], tf.string),
            'candids': tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
    
        images = tf.io.parse_tensor(features['images'], out_type=tf.float32)
        images = tf.reshape(images, shape=(3, 63, 63))
    
        object_ids = features['objectIds']
        candids = features['candids']
    
        return images, object_ids, candids
    
    def load_dataset(tfrecord_paths, batch_size):
        dataset = tf.data.TFRecordDataset(tfrecord_paths)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda images, object_ids, candids: (preprocess_image(images)[0], preprocess_image(images)[1], object_ids, candids), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    # Setup dataset
    tfrecord_folder_path = "/home/nicosepulveda/astro/bridge/TFRecords"
    all_tfrecord_paths = [str(path) for path in Path(tfrecord_folder_path).glob('*.tfrecord')]#[0:10000]
    val_dataset = load_dataset(all_tfrecord_paths, batch_size=config['batch_size'])
    
    
    # Ensure the output directory exists
    output_dir = Path("Latent_Space")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and save bottleneck features iteratively to HDF5
    def save_bottleneck_features_to_hdf5(dataset, batch_size, hdf5_file, num_batches=None):
        with h5py.File(hdf5_file, 'w') as f:
            batch_count = 0
            index = 0
            first = True
            for batch in dataset:
                inputs, _, object_ids, candids = batch
                batch_features = bottleneck_model.predict(inputs)
                object_ids_np = object_ids.numpy()  # Ensure correct type
                candids_np = candids.numpy()  # Ensure correct type
                if first:
                    max_shape = (None, batch_features.shape[1])
                    dset_features = f.create_dataset('features', data=batch_features, maxshape=max_shape, chunks=True)
                    dset_object_ids = f.create_dataset('object_ids', data=object_ids_np, maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
                    dset_candids = f.create_dataset('candids', data=candids_np, maxshape=(None,), chunks=True)
                    first = False
                else:
                    dset_features.resize(dset_features.shape[0] + batch_features.shape[0], axis=0)
                    dset_features[-batch_features.shape[0]:] = batch_features
                    dset_object_ids.resize(dset_object_ids.shape[0] + object_ids_np.shape[0], axis=0)
                    dset_object_ids[-object_ids_np.shape[0]:] = object_ids_np
                    dset_candids.resize(dset_candids.shape[0] + candids_np.shape[0], axis=0)
                    dset_candids[-candids_np.shape[0]:] = candids_np
                index += len(inputs)
                batch_count += 1
                print(f"Saved batch {batch_count}")
                if num_batches and batch_count >= num_batches:
                    break
            print(f"Finished saving {batch_count} batches.")
    
    hdf5_file = output_dir / 'val_bottleneck_features.h5'
    save_bottleneck_features_to_hdf5(val_dataset, config['batch_size'], str(hdf5_file))

if __name__ == "__main__":
    main()
