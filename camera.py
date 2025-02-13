import cv2
import numpy as np
import time
from datetime import datetime
import logging
import pytesseract
import re 
import math
from roboflow import Roboflow
import win32com.client as win32
import os

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
        self.horario_ultima_saida = None
        self.frame_anterior = None
        self.skip_frames = 50
        self.frame_count = 0 
        self.lista_deteccoes = []
        self.contador_riscos = {}
        self.tempo_minimo_nova_entrada = 60  # seconds
        self.frames_sem_movimento = 0
        self.frames_sem_pessoas = 0
        self.frames_necessarios_saida = 10  # Número de frames consecutivos necessários para confirmar saída
        self.sala_operacao = None
        self.reset_contadores()

    def reset_contadores(self):
        self.contador_riscos = {risco: 0 for risco in LISTA_RISCOS}
        self.frames_sem_movimento = 0
        self.frames_sem_pessoas = 0

    def pode_registrar_nova_entrada(self, horario_atual):
        """Verifica se já passou tempo suficiente desde a última saída"""
        if self.horario_ultima_saida is None:
            return True
        
        diferenca = (horario_atual - self.horario_ultima_saida).total_seconds()
        return diferenca >= self.tempo_minimo_nova_entrada

# Constantes
LISTA_RISCOS = ['sem máscara', 'celular', 'comida', 'sem luva', 'sem óculos', 'sem sobreposição']
DISTANCIA_MAXIMA_RISCO = 200 # pixels
TEMPO_MINIMO_SAIDA = 1  # minutos

# Inicialização do modelo Roboflow
rf = Roboflow(api_key="tVh79gxr125bhz8QJRqj")
project = rf.workspace("novamed-projeto-summer").project("inconformidades-epi-s")
modelo = project.version(1).model

# Inicialização do estado global
estado = Estado()

def envio_Email(caminhos_fotos, lista_deteccoes, horario):
    """
    Envia e-mail com detecções de risco incluindo ambas as imagens
    """
    try:
        caminho_original, caminho_anotado = caminhos_fotos
        outlook = win32.Dispatch('outlook.application')
        email = outlook.CreateItem(0)
        
        email.To = 't0047831@ems.com.br'
        email.Subject = 'Não conformidade detectada'
        email.Body = f'Olá, foi identificado as seguintes ocorrências: {lista_deteccoes} às {horario}!\n\n' \
                    f'Na sala {estado.sala_operacao}'\
                    f'Em anexo seguem duas imagens:\n' \
                    f'1. Imagem original\n' \
                    f'2. Imagem com as detecções marcadas'

        # Anexa ambas as imagens
        for caminho in [caminho_original, caminho_anotado]:
            caminho_absoluto = os.path.abspath(caminho)
            if os.path.exists(caminho_absoluto):
                email.Attachments.Add(caminho_absoluto)
            else:
                logger.error(f'Arquivo de imagem não encontrado: {caminho_absoluto}')

        email.Send()
        logger.info('E-mail enviado com sucesso!')
        return ''
    except Exception as e:
        logger.error(f'Erro ao enviar e-mail: {e}')
        return ''

def calcular_distancia(x1, y1, x2, y2):
    """Calcula a distância euclidiana entre dois pontos"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def leitura_data_hora(frame):
    """Lê a data e hora do frame usando OCR"""
    try:
        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        _, imagem_binaria = cv2.threshold(imagem_cinza, 235, 255, cv2.THRESH_BINARY)
        texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng')
        numeros = ''.join(re.findall(r'\d', texto_detectado))
        
        if len(numeros) == 14:
            try:
                data_hora = datetime.strptime(numeros, "%d%m%Y%H%M%S")
                print(data_hora)
                return data_hora
            except ValueError:
                logger.error("Erro ao converter data/hora")
                return None
        return None
    except Exception as e:
        logger.error(f"Erro na leitura de data/hora: {e}")
        return None
def leitura_sala(frame,resize_height):
    """Lê a data e hora do frame usando OCR"""
    try:
        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_start, x_end = 20, 220
        y_start, y_end = int(resize_height/1.14), resize_height-10
        imagem = frame[y_start:y_end, x_start:x_end]
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        _, imagem_binaria = cv2.threshold(imagem_cinza, 235, 255, cv2.THRESH_BINARY)
        texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng', config=config)
        print(texto_detectado)
        if texto_detectado[0:3] == 'REV' or texto_detectado[0:3] == 'REY':
            try:
                sala = texto_detectado
                if texto_detectado [0:3] == 'REY':
                    sala =  'REV ' + texto_detectado[4::]
                
                print(sala)
                return sala
            except ValueError:
                logger.error("Erro ao determinar a sala de produção")
                return None
        return None
    except Exception as e:
        logger.error(f"Erro na leitura da sala : {e}")
        return None

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
        return 0
        
    frame_diff = cv2.absdiff(frame_gray, frame_anterior)
    _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh)

def salvar_imagem_risco(frame_original, frame_anotado, risco):
    """
    Salva tanto a imagem original quanto a imagem com as anotações de detecção
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Cria diretório se não existir
        os.makedirs('Riscos/original', exist_ok=True)
        os.makedirs('Riscos/anotado', exist_ok=True)
        
        # Define os caminhos para as imagens
        nome_original = f'Riscos/original/risco_{risco}_{timestamp}.jpg'
        nome_anotado = f'Riscos/anotado/risco_{risco}_{timestamp}.jpg'
        
        # Salva as duas versões
        cv2.imwrite(nome_original, frame_original)
        cv2.imwrite(nome_anotado, frame_anotado)
        
        logger.info(f"Imagens de risco salvas: {nome_original} e {nome_anotado}")
        return nome_original, nome_anotado
    except Exception as e:
        logger.error(f"Erro ao salvar imagens de risco: {e}")
        return None, None

def processar_deteccoes(frame, dados, estado, contador_riscos, lista_deteccoes):
    """Processa as detecções do modelo e atualiza o frame"""
    try:
        pessoas = []
        riscos = []
        nome_imagem_original = None
        nome_imagem_anotada = None
        
        # Cria uma cópia do frame original antes de fazer as anotações
        frame_original = frame.copy()
        
        for pred in dados['predictions']:
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            width = int(pred['width'])
            height = int(pred['height'])
            classe = pred['class'].strip()
            print(classe)
            
            # Desenha retângulo e texto para todas as detecções
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
            cv2.putText(frame, classe, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if classe == 'Pessoa':
                pessoas.append((x, y, width, height))
            elif classe in LISTA_RISCOS:
                print('RISCO', classe)
                riscos.append((x, y, width, height, classe))
        
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

        for risco, contagem in contador_riscos.items():
            if (contagem > 1) and (risco not in lista_deteccoes):
                nome_imagem_original, nome_imagem_anotada = salvar_imagem_risco(frame_original, frame, risco)
                if nome_imagem_original and nome_imagem_anotada:
                    lista_deteccoes.append(risco)
                    logger.warning(f'Risco {risco} detectado {contagem} vezes.')
                    return True, (nome_imagem_original, nome_imagem_anotada)
                    
        return False, (None, None)
    except Exception as e:
        logger.error(f"Erro ao processar detecções: {e}")
        return False, (None, None)

def run(ip, username, password, channel=1, resize_width=1280, resize_height=720):
    """Loop principal de captura"""
    rtsp_urls = get_rtsp_urls(ip, username, password, channel)
    stream = connect(rtsp_urls)
    if stream is None:
        return
    
    recording = False
    video_writer = None
    ultima_deteccao = None
    nome_ultima_imagem = None
    
    try:
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        
        while True:
            ret, frame = stream.read()
            estado.frame_count += 1
            
            if ret:
                reconnect_attempts = 0
                frame = resize_frame(frame, resize_width, resize_height)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                x_start, x_end = int(resize_width/1.5), resize_width
                y_start, y_end = 15, 70
                frame_data = frame[y_start:y_end, x_start:x_end]
                
                if estado.frame_count % estado.skip_frames == 0:
                    white_pixels = detect_motion(frame_gray, estado.frame_anterior)
                    horario_atual = leitura_data_hora(frame_data)
                    
                    if isinstance(horario_atual, datetime):
                        # Detecção de entrada
                        if white_pixels > 1000 and not estado.movimento_entrada:
                            if estado.pode_registrar_nova_entrada(horario_atual):
                                estado.movimento_entrada = True
                                estado.horario_entrada = horario_atual
                                estado.lista_deteccoes = []
                                estado.reset_contadores()
                                logger.info(f'Nova entrada detectada: {horario_atual}')
                                print("ENTROU -------------------------------------------------------")
                                
                        
                        # Detecção de saída - Modificada
                        elif estado.movimento_entrada:
                            # Verifica movimento mínimo
                            if white_pixels < 50:  # Threshold aumentado
                                estado.frames_sem_movimento += 1
                            else:
                                estado.frames_sem_movimento = 0

                            # Verifica presença de pessoas usando o modelo
                            dados = modelo.predict(frame, confidence=0.6, overlap=0.3).json()
                            pessoas_detectadas = sum(1 for pred in dados['predictions'] if pred['class'].strip() == 'Pessoa')
                            if estado.sala_operacao == None:
                                estado.sala_operacao = leitura_sala(frame, resize_height)

                            if pessoas_detectadas == 0:
                                estado.frames_sem_pessoas += 1
                            else:
                                estado.frames_sem_pessoas = 0

                            # Confirma saída apenas se ambas as condições persistirem
                            if (estado.frames_sem_movimento >= estado.frames_necessarios_saida and 
                                estado.frames_sem_pessoas >= estado.frames_necessarios_saida):
                                if (horario_atual - estado.horario_entrada).total_seconds() > TEMPO_MINIMO_SAIDA * 60:
                                    estado.movimento_entrada = False
                                    estado.horario_ultima_saida = horario_atual
                                    logger.info(f'Saída detectada - Entrada: {estado.horario_entrada} - Saída: {horario_atual}')
                                    print('SAIU_________________________________________________________________')
                                    
                                    # Processa os riscos detectados antes de resetar
                                    if estado.lista_deteccoes and nome_ultima_imagem:
                                        envio_Email(nome_ultima_imagem, estado.lista_deteccoes, estado.horario_entrada)
                                    
                                    estado.lista_deteccoes = []
                                    estado.reset_contadores()
                    
                            # Processamento de riscos durante presença
                            if estado.movimento_entrada:
                                
                                informe_central, nome_ultima_imagem = processar_deteccoes(
                                    frame, dados, estado, estado.contador_riscos, estado.lista_deteccoes
                                )
                                if informe_central and nome_ultima_imagem:
                                    envio_Email(nome_ultima_imagem, estado.lista_deteccoes, horario_atual)
                
                estado.frame_anterior = frame_gray.copy()
                cv2.imshow('Intelbras NVD Stream', frame)
                
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
        channel = 8,  # Ajuste o número do canal conforme necessário # 1, 8 e 12 
        resize_width = 1280,  # Largura desejada
        resize_height = 720   # Altura desejada
    )