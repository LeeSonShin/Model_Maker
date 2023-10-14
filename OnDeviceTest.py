import cv2
import os
import json
import serial
import time
import random
import numpy as np
import PIL
import tensorflow as tf
import models as M

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import threading

c = ''

#쓰레드 종료용 시그널 함수
def handler(signum, frame):
     exitThread = True

def parsing_data(data):
    # 리스트 구조로 들어 왔기 때문에
    # 작업하기 편하게 스트링으로 합침
    tmp = ''.join(data)

    #출력!
    print(tmp)

#본 쓰레드
def readThread(ser):
    global line
    global exitThread

    # 쓰레드 종료될때까지 계속 돌림
    while not exitThread:
        #데이터가 있있다면
        for c in ser.read():
            #line 변수에 차곡차곡 추가하여 넣는다.
            line.append(chr(c))

            if c == 10: #라인의 끝을 만나면..
                #데이터 처리 함수로 호출
                parsing_data(line)

                #line 변수 초기화
                del line[:]                


def getSerial(ser):
    global c
    while True:
        c = ser.read(1).decode('utf-8')
        # print(c)
        # c = ser.read(1).decode('utf-8')

def ThreadTest():
    while True:
        print('testing')

if __name__ == "__main__":
    
    # UART 설정
    serial_port = '/dev/ttyACM3'  # 시리얼 포트 이름은 운영체제 및 연결된 장치에 따라 다를 수 있습니다.
    baud_rate = 115200
    ser = serial.Serial(serial_port, baud_rate)

    t = threading.Thread(target=getSerial, args=(ser, ))
    # t = threading.Thread(target=ThreadTest)
    print("Thread Start")
    t.start()
    # t.join()

    print(tf.__version__)

    import pathlib
    workspace_dir = '/home/shin/Graduation_Project/tensorflow_custom/tflite_model'
    data_dir = '/home/shin/Graduation_Project/data/12_07_05/노지 작물 해충 진단 이미지/distributed_final/cropped/train'
    data_dir = pathlib.Path(data_dir)
    print(data_dir)

    DIR = '10_14_96_area'

    batch_size = 9
    img_height = 500
    img_width = 500
    image_channel = 3
    image_mode = 'rgb'
    if(image_channel == 1):
        image_mode = 'grayscale'

    epochs = 50


    # 이미지 파일이 있는 디렉토리 경로

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

    class_names = test_ds.class_names
    class_names = ['0', '1', '2', '3', '4']
    class_num = len(class_names)
    print(class_names)

    import matplotlib.pyplot as plt

    count_total = 0
    count_answer = 0
    for images, labels in test_ds:
        for i in range(9):
            plt.figure(figsize=(10, 10))
            ax = plt.plot()
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            plt.show(block=False)
            plt.pause(7)
            print(f'answer is {class_names[labels[i]]} and result is {c}')
            if(class_names[labels[i]] == c):
                print("match!")
                count_answer += 1
            count_total += 1
            print(count_answer / count_total * 100)
            plt.close()
    
   


# # 디렉토리 내의 모든 파일 목록 가져오기
# file_list = [file for file in os.listdir(image_directory) if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg')]
# random.shuffle(file_list)

# # UART 설정
# serial_port = 'COM13'  # 시리얼 포트 이름은 운영체제 및 연결된 장치에 따라 다를 수 있습니다.
# baud_rate = 115200
# ser = serial.Serial(serial_port, baud_rate)

# try:
#     for file_name in file_list:
#         # 이미지 파일 경로
#         image_path = os.path.join(image_directory, file_name)
        
#         # 이미지 읽기
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (96, 96))
#         # 이미지 출력
#         cv2.imshow('Image', image)
#         cv2.waitKey(2000)
#         # JSON 파일 경로
#         json_path = os.path.join(image_directory, file_name.replace('.jpg', '.jpg.json').replace('.jpeg', '.jpeg.json').replace('.JPG', '.JPG.json').replace('.JPEG', '.JPEG.json'))
        
#         # JSON 파일 읽기
#         with open(json_path, 'r') as json_file:
#             print(json_path)
#             json_data = json.load(json_file)
        
#         # JSON 데이터에서 필요한 정보 추출
#         class_value = json_data["annotations"]["object"][0]["class"]
#         grow_value = json_data["annotations"]["object"][0]["grow"]
        
#         # UART 출력
#         # if class_value in [21] and grow_value == 32:
#         if grow_value == 32:
#             uart_output = "2"
#         else:
#             uart_output = "1"

#         print(uart_output)
        
#         ser.write(uart_output.encode('utf-8'))
        
#         # 키 입력 대기 (2초간)
#         cv2.waitKey(2000)
        
#         # 이미지 창 닫기
#         cv2.destroyAllWindows()

# except KeyboardInterrupt:
#     # UART 포트 닫기
#     ser.close()
#     print("Image display and UART output finished")