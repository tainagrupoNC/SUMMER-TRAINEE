import cv2
import time
import re
import os
import logging
import numpy as np
from datetime import datetime
import pytesseract
from roboflow import Roboflow
import win32com.client as win32
import os

# Configuração do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração do Tesseract
caminho = r'Tesseract/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = caminho  
config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

# Inicialização do modelo Roboflow
rf = Roboflow(api_key="zMSFgl1hKRNzI85AvY7i")
project = rf.workspace("gruponc").project("summer-trainee-eswym")
modelo = project.version(5).model

class VideoProcessor:
    def __init__(self):
        self.riscos_sala = {}
        self.dict_salas = {}
        self.dict_horarios = {}
        self.sala = None
        self.salas_ativas = set()
        self.ultimo_movimento = {}
        self.lista_riscos = ['Sem mascara', 'Celular', 'Sem luva', 'Bota']
        self.frame_counter = 0
        self.contagem_frames_sem_movimento = {}  # Para contar frames consecutivos sem movimento
    
    def inicializar_camera(self, source_path):
        cap = cv2.VideoCapture(source_path)
        return cap if cap.isOpened() else None

    def detectar_movimento(self, frame, background_subtractor):
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        mask = background_subtractor.apply(blur)
        thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None
    
    def envio_Email(self, caminhos_fotos, lista_deteccoes, horario, sala_operacao):
        """
        Envia e-mail com detecções de risco incluindo ambas as imagens
        """
        try:
            caminho_original, caminho_anotado = caminhos_fotos
            outlook = win32.Dispatch('outlook.application')
            email = outlook.CreateItem(0)
            
            email.To = 't0047831@ems.com.br'
            email.Subject = 'Relatório de Infração Detectada'
            email.Body = f"""
            Prezados (as) gestores,

            Segue em anexo o relatório de infração detectada no período de {horario.strftime('%d/%m/%Y %H:%M:%S')}, com detalhes sobre os desvios identificados nas salas produtivas.

            Resumo das Infrações:
            • Classificação da não conformidade: {', '.join(lista_deteccoes)}
            • Sala Produtiva Impactada: {sala_operacao}

            Diante dos registros, solicitamos a avaliação da ocorrência e a implementação de ações corretivas e preventivas para evitar recorrências. A equipe de Qualidade e Segurança está à disposição para suporte na definição de medidas necessárias.

            Para mais detalhes, consulte o relatório anexo.

            Prazo para ações corretivas:
            [ 24 horas - Ação Corretiva Imediata]
            [7 dias - Ação Preventiva]

            Agradecemos a colaboração de todos na manutenção da Cultura de Qualidade e de Segurança na NOVAMED.

            Atenciosamente,

            Monitoramento NOVAMED
            """

            # Anexa ambas as imagens
            for caminho in [caminho_original, caminho_anotado]:
                caminho_absoluto = os.path.abspath(caminho)
                if os.path.exists(caminho_absoluto):
                    email.Attachments.Add(caminho_absoluto)
                else:
                    print(f'Arquivo de imagem não encontrado: {caminho_absoluto}')

            email.Send()
            print('E-mail enviado com sucesso!')
            return ''
        except Exception as e:
            print(f'Erro ao enviar e-mail: {e}')
            return ''

    def processar_frame(self, channel_id, frame, background_subtractor):
        frame_original = frame.copy()
        motion_detected = self.detectar_movimento(frame, background_subtractor)
        tempo_atual = datetime.now()
        
        # Inicializar contagem se não existir
        if channel_id not in self.contagem_frames_sem_movimento:
            self.contagem_frames_sem_movimento[channel_id] = 0

        # Verificar movimento significativo
        if motion_detected is not None:
            area = cv2.contourArea(motion_detected)
            if area > 5000:  # Reduzido o threshold para melhor detecção
                self.ultimo_movimento[channel_id] = tempo_atual
                self.contagem_frames_sem_movimento[channel_id] = 0
                
                if channel_id not in self.salas_ativas:
                    logging.info(f"Ativando sala {channel_id}")
                    self.salas_ativas.add(channel_id)
                    self.riscos_sala[channel_id] = []  # Reset da lista de riscos ao ativar
            else:
                self.contagem_frames_sem_movimento[channel_id] += 1

        # Verificar inatividade
        if channel_id in self.ultimo_movimento:
            tempo_sem_movimento = (tempo_atual - self.ultimo_movimento[channel_id]).total_seconds()
            frames_sem_movimento = self.contagem_frames_sem_movimento[channel_id]
            
            # Verifica se passou 2 minutos E teve frames suficientes sem movimento
            if tempo_sem_movimento > 120 and frames_sem_movimento > 100:  # ~100 frames sem movimento
                if channel_id in self.salas_ativas:
                    logging.info(f"Desativando sala {channel_id} após {tempo_sem_movimento:.1f} segundos sem movimento")
                    self.salas_ativas.remove(channel_id)
                    self.ultimo_movimento.pop(channel_id)
                    self.riscos_sala[channel_id] = []  # Reset da lista de riscos ao desativar
                    self.contagem_frames_sem_movimento[channel_id] = 0

        # Realizar predição em salas ativas a cada 50 frames
        if self.frame_counter % 50 == 0 and channel_id in self.salas_ativas:
            logging.info(f"Realizando predição na sala {channel_id}")
            self.realizar_predicao(frame, frame_original, channel_id)
            if self.sala== None: 
                sala = self.leitura_sala(self, frame)
                if sala != None:
                    self.dict_salas[channel_id] = sala
                    print(sala)

        return frame

    def realizar_predicao(self, frame, frame_original, channel_id):
        dados = modelo.predict(frame, confidence=0.6, overlap=0.3).json()
        pessoas = []
        riscos = []

        # Processar predições
        for pred in dados['predictions']:
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            width = int(pred['width'])
            height = int(pred['height'])
            classe = pred['class'].strip()
            
            cor = (0, 255, 0)
            if classe == 'Pessoa':
                pessoas.append((x, y, width, height))
            elif classe in self.lista_riscos:
                cor = (0, 0, 255)
                riscos.append((x, y, width, height, classe))
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), cor, 1)
            cv2.putText(frame, classe, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

        # Se não houver pessoas detectadas, resetar lista de riscos
        if not pessoas and channel_id in self.riscos_sala:
            logging.info(f"Nenhuma pessoa detectada na sala {channel_id} - Resetando lista de riscos")
            self.riscos_sala[channel_id] = []
        else:
            # Associar riscos às pessoas
            self.associar_riscos_pessoas(riscos, pessoas, frame, frame_original, channel_id)
    @staticmethod
    def leitura_data_hora(self, frame):
        """Lê a data e hora do frame usando OCR"""
        try:
            x_start, x_end = int(self.resize_width/1.5), self.resize_width
            y_start, y_end = 15, 70
            frame = frame[y_start:y_end, x_start:x_end]
            imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            _, imagem_binaria = cv2.threshold(imagem_cinza, 235, 255, cv2.THRESH_BINARY)
            texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng', config=config)
            numeros = ''.join(re.findall(r'\d', texto_detectado))
            
            if len(numeros) == 14:
                try:
                    data_hora = datetime.strptime(numeros, "%d%m%Y%H%M%S")
                    logging.debug(f"Data/hora detectada: {data_hora}")
                    return data_hora
                except ValueError:
                    logging.error("Erro ao converter data/hora")
                    return None
            return None
        except Exception as e:
            logging.error(f"Erro na leitura de data/hora: {e}")
            return None
    @staticmethod
    def leitura_sala(self, frame):
        """Lê a sala do frame usando OCR"""
        try:
            imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x_start, x_end = 20, 220
            y_start, y_end = int(self.resize_height/1.14), self.resize_height-10
            imagem = frame[y_start:y_end, x_start:x_end]
            imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            _, imagem_binaria = cv2.threshold(imagem_cinza, 235, 255, cv2.THRESH_BINARY)
            texto_detectado = pytesseract.image_to_string(imagem_binaria, lang='eng', config=config)
            
            if texto_detectado[0:3] == 'REV' or texto_detectado[0:3] == 'REY':
                try:
                    sala = texto_detectado
                    if texto_detectado[0:3] == 'REY':
                        sala = 'REV ' + texto_detectado[4::]
                    return sala.strip()
                except ValueError:
                    logging.error("Erro ao determinar a sala de produção")
                    return None
            return None
        except Exception as e:
            logging.error(f"Erro na leitura da sala: {e}")
            return None

    def associar_riscos_pessoas(self, riscos, pessoas, frame, frame_original, channel_id):
        for (x_r, y_r, w_r, h_r, classe_r) in riscos:
            min_dist = float('inf')
            pessoa_associada = None

            for (x_p, y_p, w_p, h_p) in pessoas:
                centro_risco = (x_r + w_r // 2, y_r + h_r // 2)
                centro_pessoa = (x_p + w_p // 2, y_p + h_p // 2)
                dist = np.sqrt((centro_risco[0] - centro_pessoa[0])**2 + (centro_risco[1] - centro_pessoa[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    pessoa_associada = (x_p, y_p, w_p, h_p)
            
            if pessoa_associada and classe_r not in self.riscos_sala.get(channel_id, []):
                x_p, y_p, w_p, h_p = pessoa_associada
                c1, c2 = self.salvar_imagem_risco(frame, frame_original, channel_id, classe_r, x_p, y_p, w_p, h_p)
                self.envio_Email([c1, c2], self.riscos_sala[channel_id], datetime.now(), self.sala)
                if channel_id not in self.riscos_sala:
                    self.riscos_sala[channel_id] = []
                self.riscos_sala[channel_id].append(classe_r)

    def salvar_imagem_risco(self, frame_anotado, frame_original, channel_id, classe, x, y, width, height, folga=0.2):
        try:
            aspect_ratio = width / height
            new_width = width
            new_height = height

            if aspect_ratio > 1:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)

            x_start = max(x - int(width * folga), 0)
            y_start = max(y - int(height * folga), 0)
            x_end = min(x + new_width + int(width * folga), frame_anotado.shape[1])
            y_end = min(y + new_height + int(height * folga), frame_anotado.shape[0])

            pessoa_crop1 = frame_anotado[y_start:y_end, x_start:x_end]
            pessoa_crop2 = frame_original[y_start:y_end, x_start:x_end]
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            nome_imagem = f'{channel_id}_{classe}_{timestamp}.png'
            
            os.makedirs('Infra/Anotado/', exist_ok=True)
            os.makedirs('Infra/Original/', exist_ok=True)
            
            caminho_imagem1 = os.path.join('Infra/Anotado/', nome_imagem)
            caminho_imagem2 = os.path.join('Infra/Original/', nome_imagem)
            
            cv2.imwrite(caminho_imagem1, pessoa_crop1)
            cv2.imwrite(caminho_imagem2, pessoa_crop2)
            logging.info(f"Imagem salva com sucesso: {caminho_imagem1}")
        except Exception as e:
            logging.error(f"Erro ao salvar imagem: {e}")
        return caminho_imagem1 , caminho_imagem2

    def processar_videos(self, video_sources):
        background_subtractors = {
            channel_id: cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True) 
            for channel_id in video_sources
        }
        caps = {channel_id: self.inicializar_camera(source) for channel_id, source in video_sources.items()}
        windows = {channel_id: f"Channel {channel_id}" for channel_id in video_sources}
        
        for window in windows.values():
            cv2.namedWindow(window)
        
        try:
            while True:
                for channel_id, cap in caps.items():
                    if cap is None:
                        logging.error(f"Não foi possível abrir o canal {channel_id}")
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        logging.debug(f"Fim do vídeo no canal {channel_id}")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    processed_frame = self.processar_frame(channel_id, frame, background_subtractors[channel_id])
                    cv2.imshow(windows[channel_id], processed_frame)
                    self.frame_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            for cap in caps.values():
                if cap:
                    cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_sources = {
        1: "Pasta_Videos/SALA_REV.mp4",
        2: "Pasta_Videos/SALA_REV_500.mp4"
    }
    processor = VideoProcessor()
    processor.processar_videos(video_sources)