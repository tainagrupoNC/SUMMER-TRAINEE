import pytesseract
from PIL import Image
import cv2 

# Carregando a imagem
imagem_path = 'image_code.jpg'  # Caminho no qual a imagem está localizada
imagem = cv2.imread(imagem_path) # Imagem a ser utilizada carregada 

# Carregando a extensão Tesseract 
caminho = 'Tesseract/tesseract.exe' # Caminho do executável 
pytesseract.pytesseract.tesseract_cmd = caminho # Executando a biblioteca 
img = Image.open(imagem_path)  # Abertura da imagem 
text = pytesseract.image_to_string(img) # Conversão de imagem para string 
print(text) # Print da mensagem/ código detectado 

# Ilustra na tela a imagem em análise 
cv2.imshow('Imagem de Exemplo', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()