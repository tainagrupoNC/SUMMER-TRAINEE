from datetime import datetime
import pytesseract
import cv2
import re 
# Defina o caminho correto para o executável do Tesseract
caminho = r'Tesseract/tesseract.exe'  
pytesseract.pytesseract.tesseract_cmd = caminho  
config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'


def leitura_data_hora(frame):
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convertendo a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicando binarização (thresholding)
    _, imagem_binaria = cv2.threshold(imagem_cinza, 210, 255, cv2.THRESH_BINARY)

    # Aplicando um filtro para remover ruídos (opcional)
    imagem_sem_ruido = cv2.medianBlur(imagem_binaria, 5)

    # Ajustando o contraste (opcional)
    alpha = 1 # Fator de contraste
    beta = 50    # Fator de brilho
    imagem_processada = cv2.convertScaleAbs(imagem_sem_ruido, alpha=alpha, beta=beta)
    #cv2.imshow('Imagem processada', imagem_processada)
    #cv2.waitKey(0)
    texto_detectado = pytesseract.image_to_string(imagem_processada, lang='eng')  # Use o idioma adequado
    numeros = ''.join(re.findall(r'\d', texto_detectado))
    if len(numeros) == 14: 
        data_hora = datetime.strptime(numeros, "%d%m%Y%H%M%S")
        return data_hora
    return 'Data não detectada verifique a imagem'


#print(leitura_data_hora(cv2.imread('Images/img_data.jpg')))