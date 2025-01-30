from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import numpy
from IPython.display import clear_output

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

############################################################
cap = cv2.VideoCapture("SALA_REV.mp4")
frame_atual = 1
ret, frame = cap.read()

modelo = YOLO('best.pt')


def yolo_infos(frame):
    # Carregar o modelo YOLO
    model = YOLO("best.pt")

    # Configurações
    CONF = 0.53  # Confiança mínima
    DIR = frame  # Caminho do arquivo de entrada
    SAVE = True  # Não salvar os resultados
    NAME = "foto_cropada"  # Nome do arquivo salvo (se SAVE=True)

    # Executar o modelo
    results = model(source=DIR, conf=CONF, save=SAVE, name=NAME)

    # Obter as informações do bounding box e as classificações
    detections = []
    for result in results:
        boxes = result.boxes  # Obter as caixas de detecção
        class_ids = boxes.cls.cpu().numpy()  # IDs das classes detectadas
        scores = boxes.conf.cpu().numpy()  # Confiança das detecções
        bboxes = boxes.xyxy.cpu().numpy()  # Coordenadas dos bounding boxes

        # Mapear IDs para nomes das classes e coletar bounding boxes
        for cls_id, score, bbox in zip(class_ids, scores, bboxes):
            class_name = model.names[int(cls_id)]  # Nome da classe
            detections.append({
                "class": class_name,
                "confidence": score,
                "bbox": bbox
            })

    # Exibir as detecções com bounding boxes
    #print("Detecções realizadas:")
    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        print(f"Classe: {class_name}, Confiança: {confidence:.2f}, Bounding Box: {bbox}")
    return ''

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while frame_atual < num_frames:
    ret, frame = cap.read()
    if frame_atual % 10 == 0:
        yolo_infos(frame)
    cv2.imshow('Video', frame)
    frame_atual += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('fim')