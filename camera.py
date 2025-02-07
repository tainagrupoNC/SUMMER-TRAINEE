import cv2
import numpy as np
import time
from datetime import datetime
import logging
import pytesseract
import re 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Defina o caminho correto para o executável do Tesseract
caminho = r'Tesseract/tesseract.exe'  
pytesseract.pytesseract.tesseract_cmd = caminho  
config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'


def leitura_data_hora(frame):
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convertendo a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Aplicando binarização (thresholding)
    _, imagem_binaria = cv2.threshold(imagem_cinza,235, 255, cv2.THRESH_BINARY) #220
    # Aplicando um filtro para remover ruídos (opcional)
    #imagem_sem_ruido = cv2.medianBlur(imagem_binaria, 5)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #imagem_sem_ruido = cv2.morphologyEx(imagem_binaria, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('Imagem processada', imagem_sem_ruido)
    #cv2.waitKey(0)
    texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng')  # Use o idioma adequado
    numeros = ''.join(re.findall(r'\d', texto_detectado))
    if len(numeros) == 14: 
        data_hora = datetime.strptime(numeros, "%d%m%Y%H%M%S")
        return data_hora
    return 'Data não detectada verifique a imagem'''

def get_rtsp_urls(ip, username, password, channel):
    return [
        f'rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0',  # Main stream
        f'rtsp://{username}:{password}@{ip}/h264/ch{channel}/main/av_stream',  # Alternativa 1
        f'rtsp://{username}:{password}@{ip}/streaming/channels/{channel}01',   # Alternativa 2
        f'rtsp://{username}:{password}@{ip}/profile1/media.smp',              # Alternativa 3
        f'rtsp://{username}:{password}@{ip}/live/main',                       # Alternativa 4
    ]

def connect(rtsp_urls):
    """Tenta conectar usando diferentes URLs RTSP"""
    for url in rtsp_urls:
        logger.info(f"Tentando conectar via: {url}")
        
        # Configura o pipeline RTSP com buffer mínimo
        stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if stream.isOpened():
            # Configurações para melhor performance
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
        frame = resize_frame(frame, width, height)  # Redimensiona antes de configurar o VideoWriter
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
            
            if ret:
                reconnect_attempts = 0  # Reset contador de reconexão
                
                # Redimensiona o frame
                frame = resize_frame(frame, resize_width, resize_height)
                x_start, x_end = int(resize_width/1.5), resize_width
                y_start, y_end = 15, 70
                frame_data = frame[y_start:y_end, x_start:x_end]
                print(leitura_data_hora(frame_data))
                
                # Mostra o frame
                cv2.imshow('Intelbras NVD Stream', frame)
                
                # Grava se estiver em modo de gravação
                if recording and video_writer is not None:
                    video_writer.write(frame)
                
                # Controles
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
                    # Salva snapshot
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
        ip="10.4.28.16",
        username="usuario",
        password="Nmed@2025",
        channel=8,  # Ajuste o número do canal conforme necessário
        resize_width=1280,  # Largura desejada
        resize_height=720   # Altura desejada
    )