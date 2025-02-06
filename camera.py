from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from PIL import ImageGrab
import pyautogui
import cv2 
import numpy as np
import time
import mss


# Baixa e gerencia a versão correta do WebDriver automaticamente
service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service)

driver.get("http://10.4.28.16")
####################################################################################################################
wait = WebDriverWait(driver, 10)  # Espera até 10 segundos
username = wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div/div/div/div/div/div/div/table[1]/tbody/tr/td[2]/input')))
password = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div/div/div/div/div/div/table[5]/tbody/tr/td[2]/table/tbody/tr/td[1]/input")))
username.send_keys("usuario")
password.send_keys("Nmed@2025")
login_button = wait.until(EC.presence_of_element_located((By.ID, 'loginButton')))
login_button.click()
camera_unica_button = wait.until(EC.presence_of_element_located((By.ID,'button-1234-btnIconEl')))
camera_unica_button.click()
selecao_camera = wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[8]/div[1]/div[1]/div/div/div/div/div/div[3]/div/div/div/div/div/span/div/a[8]/span/span')))
selecao_camera.click()
'''button_split = wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[8]/div[1]/div[1]/div/div/div/div/div/div[3]/div/div/div/div/div/span/div/a[1]/span/span')))
button_split.click'''
pyautogui.press('f11')
time.sleep(2)
actions = ActionChains(driver)
# Obtenha a resolução da janela
window_width = driver.execute_script("return window.innerWidth")
window_height = driver.execute_script("return window.innerHeight")
# Calcule as coordenadas do centro da tela
center_x = window_width // 2
center_y = window_height // 2
actions.move_by_offset(center_x, center_y).double_click().perform()
time.sleep(1)  # Intervalo entre os cliques duplos
actions.move_by_offset(-center_x, -center_y)  # Resetar a posição do cursor


# Configuração do mss para captura de tela
sct = mss.mss()

# Obtendo as coordenadas da janela do navegador
window_rect = driver.get_window_rect()
x, y, w, h = window_rect['x'], window_rect['y'], window_rect['width'], window_rect['height']

# Laço para captura contínua de tela
while True:
    # Captura da área da janela do navegador
    screenshot = sct.grab({"top": y, "left": x, "width": w, "height": h})
    
    # Convertendo para array numpy
    frame = np.array(screenshot)

    frame_rgb = frame[:, :, [2, 1, 0, 3]]

    # Exibindo o frame capturado
    cv2.imshow('Live Feed', frame_rgb)

    # Condição para sair do loop (pressionar 'q' no teclado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar tudo
cv2.destroyAllWindows()
driver.quit()