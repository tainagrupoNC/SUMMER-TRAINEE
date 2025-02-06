import cv2
from roboflow import Roboflow
from Data import leitura_data_hora
import datetime
import math

# Inicialização da câmera/vídeo
camera_url = "SALA_REV.mp4"
cap = cv2.VideoCapture(camera_url)

ret, frame = cap.read()
if not ret or frame is None:
    print("Erro ao carregar o vídeo.")
    exit()

def calcular_distancia(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# Inicialização do modelo Roboflow
rf = Roboflow(api_key="zMSFgl1hKRNzI85AvY7i")
project = rf.workspace("gruponc").project("summer-trainee-eswym")
modelo = project.version(5).model

# Reduzir a resolução do frame para acelerar o processamento
frame_width = 640
frame_height = 480

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames
frame = cv2.resize(frame, (frame_width, frame_height))  # Redimensionamento do frame
frame_anterior = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversão para escala de cinza

# Variáveis de controle
movimento_entrada = False
frame_atual = 1  # Contador de frames
informe_central = False

# Lista de atos contra a política de segurança
lista_riscos = ['Sem mascara', 'Celular', 'Alimento', 'Sem luva', 'Sem óculos']

# Contador de riscos
contador_riscos = {risco: 0 for risco in lista_riscos}

while frame_atual < num_frames:
    ret, frame_i = cap.read()  # Coleta dos frames presentes no vídeo
    if not ret or frame_i is None:  # Sai do loop se não houver mais frames
        break

    frame = cv2.resize(frame_i, (frame_width, frame_height))  # Redimensionamento do frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversão do frame para escala de cinza

    # Executar ações a cada 50 frames para deixar o processamento mais rápido (cerca de 1 frame por segundo)
    if frame_atual % 50 == 0:
        frame_resize = frame_i[20:100, 850:1250]  # Recorte da imagem
        frame_diff = cv2.absdiff(frame_gray, frame_anterior)  # Cálculo da diferença entre frames
        _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)  # Limiarização da diferença
        white_pixels = cv2.countNonZero(thresh)  # Contagem de pixels brancos
        if white_pixels > 500:
            frame_entrada = frame_i 

        if white_pixels > 500 and not movimento_entrada:  # Se houver movimento
            horario_entrada = leitura_data_hora(frame_resize)  # Leitura da data e hora de entrada (detecção de movimento)
            if isinstance(horario_entrada, datetime.datetime):  # Verificação do type da saída
                movimento_entrada = True  # Atualização da variável de controle
            else:
                continue

        # Se não for mais detectado movimento e não houver mais pixels brancos
        if movimento_entrada and white_pixels < 1:
            horario_atual = leitura_data_hora(frame_resize)  # Horário atual de acordo com a filmagem
            frame_resize = frame_entrada[20:100, 850:1250]
            horario_intermediario = leitura_data_hora(frame_resize)
            if isinstance(horario_intermediario, datetime.datetime):
                if isinstance(horario_atual, datetime.datetime):  # Verificação do tipo se corresponde a data e horário
                    if (horario_atual - horario_intermediario).total_seconds() / 60 > 0.5:
                        horario_saida = horario_atual
                        movimento_entrada = False
                        informe_central = False
                        contagem = 0
                        contador_riscos = {risco: 0 for risco in lista_riscos}  # Reiniciar o contador de riscos
                        print(f'Entrada: {horario_entrada} - Saída: {horario_saida}')
                        print('SEM PESSOAS NA SALA')

        if movimento_entrada and not informe_central:
            dados = modelo.predict(frame, confidence=0.6, overlap=0.3).json()  # Predição do modelo
            pessoas = []
            riscos = []
            # Desenhar as bounding boxes no frame
            for pred in dados['predictions']:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                width = int(pred['width'])
                height = int(pred['height'])
                classe = pred['class'].strip()  # Remove espaços extras
                ########################3
                if classe == 'Pessoa':
                    pessoas.append((x,y,width, height))
                elif classe in lista_riscos:
                    riscos.append((x,y,width,height, classe))
                cor = (0, 255, 0)
    
                # Desenhar a bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), cor, 1)
                # Adicionar o texto da classe
                cv2.putText(frame, classe, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)


            for (x_p, y_p, w_p, h_p) in pessoas:
                for (x_r, y_r, w_r, h_r, classe_r) in riscos:
                    centro_pessoa = (x_p + w_p // 2, y_p + h_p // 2)
                    centro_risco = (x_r + w_r // 2, y_r + h_r // 2)
                    distancia = calcular_distancia(centro_pessoa[0], centro_pessoa[1], centro_risco[0], centro_risco[1])

                    if distancia < 100:  # Defina o valor de proximidade desejado
                        print(f'Risco {classe_r} detectado próximo a uma pessoa.')
                        contador_riscos[classe_r] = contador_riscos.get(classe_r, 0) + 1
                        cor = (0, 0, 255)  # Vermelho para riscos
                         # Desenhar a bounding box
                        cv2.rectangle(frame, (x_r, y_r), (x_r + w_r, y_r + h_r), cor, 1)
                        # Adicionar o texto da classe
                        cv2.putText(frame, classe, (x_r, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                            

            # Verificação dos riscos detectados mais de uma vez
            for risco, contagem in contador_riscos.items():
                if contagem > 1:
                    cv2.imwrite(f'Riscos/{frame_atual}.jpg', frame)
                    print(f'Risco {risco} detectado mais de uma vez. A pessoa deve sair da sala.')
                    informe_central = True

    # Exibição do vídeo
    cv2.imshow('Classificação', frame)
    frame_anterior = frame_gray.copy()
    frame_atual += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
