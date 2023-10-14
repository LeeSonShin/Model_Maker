import numpy as np
import PIL
import tensorflow as tf
import models as M

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.__version__)

import pathlib
workspace_dir = '/home/shin/Graduation_Project/tensorflow_custom/tflite_model'
data_dir = '/home/shin/Graduation_Project/data/12_07_05/노지 작물 해충 진단 이미지/distributed_final/cropped/train'
data_dir = pathlib.Path(data_dir)
print(data_dir)

DIR = '10_14_96_area'
#DIR = 'custom_cnn'
batch_size = 32
img_height = 96
img_width = 96
image_channel = 3
image_mode = 'rgb'
if(image_channel == 1):
   image_mode = 'grayscale'

epochs = 100


image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.JPG'))) +len(list(data_dir.glob('*/*.JPEG'))) + len(list(data_dir.glob('*/*.jpeg')))  
print(image_count)



train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="both",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = image_mode,
  crop_to_aspect_ratio = True,
  interpolation='area',
  )

class_names = train_ds.class_names
class_num = len(class_names)
print(class_names)


test_dir = '/home/shin/Graduation_Project/data/12_07_05/노지 작물 해충 진단 이미지/distributed_final/cropped/test'
test_ds= tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = image_mode,
  crop_to_aspect_ratio = True,
  interpolation='area',
  )


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  image_channel)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = M.Mobilenet_v3(img_width,img_height,image_channel,class_num)
# model = M.cnn_example(img_width,img_height,image_channel,class_num)
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs=epochs
)

model.evaluate(
    train_ds
)

export_dir = f'{workspace_dir}/saved_model/{DIR}'
tf.saved_model.save(model, export_dir)

# Convert the model.
tflite_model_path = f'{workspace_dir}/saved_model/{DIR}/saved_model.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(tflite_model_path, 'wb') as f:
  f.write(tflite_model)

TF_MODEL_FILE_PATH = tflite_model_path # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)


numpy_arrays = []

for batch in train_ds:
    images, labels = batch
    for image in images:
      image_numpy = image.numpy()  # 이미지 배치를 NumPy 배열로 변환

      numpy_arrays.append(image_numpy)

print(len(numpy_arrays))
print(len(numpy_arrays[0]))
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(numpy_arrays).batch(1).take(100):
    yield [input_value]



converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_model_quant = converter.convert()

tflite_model_quant_path = f'{workspace_dir}/saved_model/{DIR}/saved_model_quant.tflite'
with open(tflite_model_quant_path, 'wb') as f:
    f.write(tflite_model_quant)

print(f"TensorFlow Lite 모델이 '{tflite_model_quant_path}'에 저장되었습니다.")
