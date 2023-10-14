import numpy as np
import tensorflow as tf

import pathlib
workspace_dir = '/home/shin/Graduation_Project/tensorflow_custom/tflite_model'
data_dir = '/home/shin/Graduation_Project/data/12_07_05/노지 작물 해충 진단 이미지/distributed_final/cropped/train'
data_dir = pathlib.Path(data_dir)
print(data_dir)

DIR = '10_14'

batch_size = 32
img_height = 96
img_width = 96

image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.JPG'))) +len(list(data_dir.glob('*/*.JPEG'))) + len(list(data_dir.glob('*/*.jpeg')))  
print(image_count)


test_dir = '/home/shin/Graduation_Project/data/12_07_05/노지 작물 해충 진단 이미지/distributed_final/cropped/test'
test_ds= tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='grayscale'
  )




# TFLite 모델 파일 경로 설정
tflite_model_quant_path = f'{workspace_dir}/saved_model/{DIR}/saved_model_quant.tflite'

# TFLite 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path=tflite_model_quant_path)
interpreter.allocate_tensors()

# 입력 및 출력 텐서의 인덱스 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# 정확도를 계산할 변수 초기화
correct_predictions = 0
total_predictions = 0

# 각 이미지에 대한 추론 실행 및 정확도 계산
for images, labels in test_ds:
    for i in range(len(images)):
        # 이미지를 TFLite 모델에 입력
        input_data = np.expand_dims(images[i], axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 모델 실행
        interpreter.invoke()

        # 모델 출력 가져오기
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 추론 결과에서 클래스 선택 (예시: 가장 높은 확률의 클래스 선택)
        predicted_class = np.argmax(output_data)

        # 실제 레이블과 비교하여 정확하게 분류된 경우 카운트
        if predicted_class == labels[i].numpy():
            correct_predictions += 1

        total_predictions += 1

# 정확도 계산
accuracy = correct_predictions / total_predictions
print("정확도:", accuracy)