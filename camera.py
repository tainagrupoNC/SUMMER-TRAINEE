import cv2
import numpy as np
import time
from datetime import datetime
import logging
import pytesseract
import re 
import math
from roboflow import Roboflow

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuração do Tesseract
caminho = r'Tesseract/tesseract.exe'  
pytesseract.pytesseract.tesseract_cmd = caminho  
config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

# Variáveis globais de estado
class Estado:
    def __init__(self):
        self.movimento_entrada = False
        self.informe_central = False
        self.horario_entrada = None
        self.frame_anterior = None
        self.skip_frames = 50
        self.frame_count = 0 
        self.informe_central = False
        self.contador_riscos = {}
        self.reset_contadores()

    def reset_contadores(self):
        self.contador_riscos = {risco: 0 for risco in LISTA_RISCOS}

# Constantes
LISTA_RISCOS = ['Sem mascara', 'Celular', 'Alimento', 'Sem luva', 'Sem óculos']
DISTANCIA_MAXIMA_RISCO = 100  # pixels
TEMPO_MINIMO_SAIDA = 1  # minutos

# Inicialização do modelo Roboflow
rf = Roboflow(api_key="zMSFgl1hKRNzI85AvY7i")
project = rf.workspace("gruponc").project("summer-trainee-eswym")
modelo = project.version(5).model

# Inicialização do estado global
estado = Estado()

def calcular_distancia(x1, y1, x2, y2):
    """Calcula a distância euclidiana entre dois pontos"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def leitura_data_hora(frame):
    """Lê a data e hora do frame usando OCR"""
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, imagem_binaria = cv2.threshold(imagem_cinza, 235, 255, cv2.THRESH_BINARY)
    texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng')
    numeros = ''.join(re.findall(r'\d', texto_detectado))
    if len(numeros) == 14:
        try:
            data_hora = datetime.strptime(numeros, "%d%m%Y%H%M%S")
            return data_hora
        except ValueError:
            logger.error("Erro ao converter data/hora")
            return 'Data inválida'
    return 'Data não detectada verifique a imagem'

def get_rtsp_urls(ip, username, password, channel):
    """Retorna lista de possíveis URLs RTSP"""
    return [
        f'rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0',
        f'rtsp://{username}:{password}@{ip}/h264/ch{channel}/main/av_stream',
        f'rtsp://{username}:{password}@{ip}/streaming/channels/{channel}01',
        f'rtsp://{username}:{password}@{ip}/profile1/media.smp',
        f'rtsp://{username}:{password}@{ip}/live/main',
    ]

def connect(rtsp_urls):
    """Tenta conectar usando diferentes URLs RTSP"""
    for url in rtsp_urls:
        logger.info(f"Tentando conectar via: {url}")
        stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if stream.isOpened():
            stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            stream.set(cv2.CAP_PROP_FPS, 30)
            stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            logger.info(f"Conectado com sucesso via {url}")
            return stream
        stream.release()
    
    logger.error("Não foi possível conectar a nenhuma URL RTSP")
    return None

def resize_frame(frame, width, height):
    """Redimensiona o frame para o tamanho especificado"""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def start_recording(stream, width, height, output_path=None):
    """Inicia gravação do vídeo"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'intelbras_recording_{timestamp}.mp4'
    
    ret, frame = stream.read()
    if ret:
        frame = resize_frame(frame, width, height)
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30.0,
            (width, height)
        )
        logger.info(f"Gravação iniciada: {output_path}")
        return video_writer
    return None

def stop_recording(video_writer):
    """Para a gravação"""
    if video_writer is not None:
        video_writer.release()
        logger.info("Gravação finalizada")

def detect_motion(frame_gray, frame_anterior):
    """Detecta movimento significativo entre frames"""
    if frame_anterior is None:
        return False
        
    frame_diff = cv2.absdiff(frame_gray, frame_anterior)
    _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)  # Limiarização da diferença
    white_pixels = cv2.countNonZero(thresh)
    return white_pixels


def salvar_imagem_risco(frame, risco):
    """Salva imagem quando um risco é detectado"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f'Riscos/risco_{risco}_{timestamp}.jpg'
    cv2.imwrite(nome_arquivo, frame)
    logger.info(f"Imagem de risco salva: {nome_arquivo}")

def processar_deteccoes(frame, dados, estado):
    """Processa as detecções do modelo e atualiza o frame"""
    pessoas = []
    riscos = []
    
    for pred in dados['predictions']:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        width = int(pred['width'])
        height = int(pred['height'])
        classe = pred['class'].strip()
        
        if classe == 'Pessoa':
            pessoas.append((x, y, width, height))
        elif classe in LISTA_RISCOS:
            print('RISCO: ', classe)
            riscos.append((x, y, width, height, classe))
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.putText(frame, classe, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    for pessoa in pessoas:
        x_p, y_p, w_p, h_p = pessoa
        centro_pessoa = (x_p + w_p // 2, y_p + h_p // 2)
        
        for risco in riscos:
            x_r, y_r, w_r, h_r, classe_r = risco
            centro_risco = (x_r + w_r // 2, y_r + h_r // 2)
            distancia = calcular_distancia(centro_pessoa[0], centro_pessoa[1], 
                                         centro_risco[0], centro_risco[1])
            
            if distancia < DISTANCIA_MAXIMA_RISCO:
                estado.contador_riscos[classe_r] += 1
                cv2.rectangle(frame, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 0, 255), 1)
                cv2.putText(frame, classe_r, (x_r, y_r - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                logger.warning(f'Risco {classe_r} detectado próximo a uma pessoa.')

    for risco, contagem in estado.contador_riscos.items():
        if contagem > 1:
            salvar_imagem_risco(frame, risco)
            logger.warning(f'Risco {risco} detectado {contagem} vezes.')
            return True
    return False

def cleanup(stream, recording, video_writer):
    """Limpa recursos"""
    if recording:
        stop_recording(video_writer)
    if stream is not None:
        stream.release()
    cv2.destroyAllWindows()
    logger.info("Recursos liberados")

def run(ip, username, password, channel=1, resize_width=1280, resize_height=720):
    """Loop principal de captura"""
    rtsp_urls = get_rtsp_urls(ip, username, password, channel)
    stream = connect(rtsp_urls)
    if stream is None:
        return
    
    recording = False
    video_writer = None
    
    try:
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        
        while True:
            ret, frame = stream.read()
            estado.frame_count+=1
            
            if ret:
                reconnect_attempts = 0
                frame = resize_frame(frame, resize_width, resize_height)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                x_start, x_end = int(resize_width/1.5), resize_width
                y_start, y_end = 15, 70
                frame_data = frame[y_start:y_end, x_start:x_end]
                if estado.frame_count % estado.skip_frames == 0:
                    white_pixels = detect_motion(frame_gray, estado.frame_anterior) # Contagem de pixels brancos
                    print("WHITE PIXELS: ", white_pixels)
                    if white_pixels > 1000 and not estado.movimento_entrada: 
                        horario_entrada = leitura_data_hora(frame_data)
                        if isinstance(horario_entrada, datetime) and not estado.movimento_entrada:
                            estado.movimento_entrada = True
                            print('ENTRADA: ', horario_entrada)
                    
                    if white_pixels < 100 and estado.movimento_entrada == True:
                        horario_atual = leitura_data_hora(frame_data)
                        if isinstance(horario_atual, datetime):
                            if (horario_atual - horario_entrada).total_seconds() / 60 > 1:
                                horario_saida = horario_atual
                                print(f'Entrada: {horario_entrada} - Saída: {horario_saida}')
                                print('SEM PESSOAS NA SALA')
                                estado.movimento_entrada = False
                                estado.informe_central = False
                                estado.contador_riscos = {}  # Reiniciar o contador de riscos

                    if estado.movimento_entrada and not estado.informe_central:
                        dados = modelo.predict(frame, confidence=0.6, overlap=0.3).json()  # Predição do modelo
                        estado.informe_central = processar_deteccoes(frame, dados, estado)
           
                estado.frame_anterior = frame_gray.copy()
                cv2.imshow('Intelbras NVD Stream', frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording:
                        video_writer = start_recording(stream, resize_width, resize_height)
                        recording = True
                    else:
                        stop_recording(video_writer)
                        recording = False
                        video_writer = None
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'intelbras_snapshot_{timestamp}.jpg', frame)
                    logger.info(f"Snapshot salvo: intelbras_snapshot_{timestamp}.jpg")
            
            else:
                logger.warning("Falha ao ler frame")
                reconnect_attempts += 1
                
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error("Muitas falhas consecutivas, tentando reconectar...")
                    stream.release()
                    stream = connect(rtsp_urls)
                    if stream is None:
                        break
                    reconnect_attempts = 0
                
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Programa interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante a captura: {e}")
    finally:
        cleanup(stream, recording, video_writer)

def cleanup(stream, recording, video_writer):
    """Limpa recursos"""
    if recording:
        stop_recording(video_writer)
    if stream is not None:
        stream.release()
    cv2.destroyAllWindows()
    logger.info("Recursos liberados")

if __name__ == "__main__":
    # Exemplo de uso
    run(
        ip = "10.4.28.16",
        username = "usuario",
        password = "Nmed@2025",
        channel = 1,  # Ajuste o número do canal conforme necessário # 1, 8 e 12 
        resize_width = 1280,  # Largura desejada
        resize_height = 720   # Altura desejada
    )