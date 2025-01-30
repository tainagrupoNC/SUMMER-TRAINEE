from roboflow import Roboflow
from ultralytics import YOLO
import cv2

rf = Roboflow(api_key="zMSFgl1hKRNzI85AvY7i")
project = rf.workspace("gruponc").project("summer-trainee-project")
model = project.version(1).model

############################################################
cap = cv2.VideoCapture("SALA_REV.mp4")
frame_atual = 1
ret, frame = cap.read()

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while frame_atual < num_frames:
    ret, frame = cap.read()
    if frame_atual % 50 == 0: 
        dados = model.predict(frame, confidence=70, overlap=30).json()
        for i in range(len(dados['predictions'])):
            print(dados['predictions'][i]['class'])
        print('----------------------------')
    cv2.imshow('Video', frame)
    frame_atual += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('fim')