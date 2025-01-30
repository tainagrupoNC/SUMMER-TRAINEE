import cv2 
from ultralytics import YOLO
from celulares import leitura_celulares
import matplotlib.pyplot as plt
from Data import leitura_data_hora

cap = cv2.VideoCapture("SALA_REV.mp4")
ret, frame = cap.read()
modelo = YOLO("yolov8n.pt")
frame_atual = 1
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while frame_atual<num_frames: 
    ret, frame = cap.read()
    if frame_atual % 50 == 0: 
        celular = leitura_celulares(modelo, frame)
        frame_resize = frame[20:100, 850:1250]
        if celular:
            #cv2.imwrite('Images/img_data.jpg',frame_resize)
            print(leitura_data_hora(frame_resize))
          
    cv2.imshow('Classificação',frame)
    frame_atual+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break