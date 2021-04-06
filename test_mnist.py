#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.list_physical_devices('GPU')
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

tfds.disable_progress_bar()


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

batch_size = 2**14

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(batch_size)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(batch_size)

class BiasAdd(tf.keras.layers.Layer):
    def __init__(self, input_dim, name):
        super(BiasAdd, self).__init__(name=name)
        self.input_dim = input_dim
        b_init = tf.random_uniform_initializer(
            minval=-1.0,
            maxval=1.0,
            seed=1234567,
        )
        self.bs = [
          tf.Variable(initial_value=b_init(shape=(self.input_dim,), dtype="float32"), trainable=False)
          for i in range(0, 30)
        ]

    def call(self, inputs):
        return tf.math.add_n([tf.nn.bias_add(inputs, b) for b in self.bs])

    def get_config(self):
      return {"input_dim": self.input_dim}

# Will use the optimized kernel.
assert (28 * 28 * 1 % 4 == 0)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
] + [
  BiasAdd(28 * 28 * 1, name="bias" + str(i)) for i in range(0, 30)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Create a TensorBoard callback
logs = "logs/nvvp/baseline"

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 # histogram_freq = 1,
                                                 # profile_batch = '1,100000'
                                                 profile_batch = 0,
                                                 )

model.fit(ds_train,
          epochs=2,
          validation_data=ds_test,
          callbacks = [tboard_callback]
          )

