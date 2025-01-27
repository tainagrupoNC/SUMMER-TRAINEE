import pandas as pd
import matplotlib.pyplot as plt 
import selenium 
import cv2 

# Carregando a imagem
imagem_path = 'image_code.jpg'  # Substitua pelo caminho da sua imagem
imagem = cv2.imread(imagem_path)

# Verificando se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
else:
    print("Imagem carregada com sucesso.")


cv2.imshow('Imagem de Exemplo', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()