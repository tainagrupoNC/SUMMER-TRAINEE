
def leitura_celulares(modelo, frame):
    resultado = modelo.predict(frame, conf=0.5)
    lista = resultado[0].boxes.xyxy
    lista = lista[0]
    x = int(lista[0])
    y = int(lista[1])
    w = int(lista[2])
    h = int(lista[3])
    cls = resultado[0].boxes.cls
    if 67 in cls:
        #cv2.rectangle(frame,(x,y),(w,h),(255,0,255),5)
        #cvzone.putTextRect(frame,"CELULAR IDENTIFICADO",(105,65),colorR=(0,0,255))
        return True
    return False