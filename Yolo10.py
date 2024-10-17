import os
import shutil
import numpy as np

base_dir = 'C:\\Users\\user\\Desktop\\dataset'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

os.makedirs(os.path.join(image_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(image_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(label_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(label_dir, 'val'), exist_ok=True)

categories = ['CPU', 'GPU', 'MAINBOARD', 'RAM']

train_ratio = 0.8

for category in categories:
    category_path = os.path.join(base_dir, category)

    image_files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    np.random.shuffle(image_files)

    train_size = int(len(image_files) * train_ratio)

    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    for image_file in train_files:
        shutil.copy(os.path.join(category_path, image_file), os.path.join(image_dir, 'train', image_file))
        shutil.copy(os.path.join(category_path, image_file.replace('.jpg', '.txt')), os.path.join(label_dir, 'train', image_file.replace('.jpg', '.txt')))

    for image_file in val_files:
        shutil.copy(os.path.join(category_path, image_file), os.path.join(image_dir, 'val', image_file))
        shutil.copy(os.path.join(category_path, image_file.replace('.jpg', '.txt')), os.path.join(label_dir, 'val', image_file.replace('.jpg', '.txt')))

yaml_content = """
train: C:\\Users\\user\\Desktop\\dataset\\images\\train
val: C:\\Users\\user\\Desktop\\dataset\\images\\val

nc: 4  # 클래스 수
names: ['CPU', 'GPU', 'MAINBOARD', 'RAM']  # 클래스 이름
"""

# YAML 파일 생성
with open('C:\\Users\\user\\Desktop\\dataset\\dataset.yaml', 'w') as f:
    f.write(yaml_content)

from ultralytics import YOLO

# YOLOv10 모델 로드 (사전 훈련된 모델을 자동으로 다운로드)
model = YOLO('yolov10n.pt')  # 작은 모델을 사용, 필요에 따라 다른 모델로 변경 가능

# YOLOv10 모델 훈련
model.train(data='C:\\Users\\user\\Desktop\\dataset\\dataset.yaml', epochs=1)

# 객체 인식 수행
results = model.predict(source=os.path.join(image_dir, 'val'))  # 검증 데이터에 대해 예측

import matplotlib.pyplot as plt
import cv2

# 객체 인식 수행
results = model.predict(source=os.path.join(image_dir, 'val'))  # 검증 데이터에 대해 예측

# 결과 출력
for result in results:
    # 예측된 이미지 가져오기
    img = result.plot()  # 이미지에 결과를 겹쳐서 표시

    # matplotlib를 사용하여 이미지 출력
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 축 숨기기
    plt.show()  # 이미지 표시

# YOLOv10 모델을 ONNX 형식으로 내보내기
model.export(format='onnx', imgsz=640)  # 'best.pt'는 훈련된 모델 파일