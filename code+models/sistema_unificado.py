import cv2
import pickle
import time
import serial  # Comunicação com o Arduino
import threading
import numpy as np
import mediapipe as mp
from pathlib import Path
from ultralytics import YOLO

# ==========================================================
# CONFIGURAÇÕES DA SERIAL E PORTAS (Ajuste se necessário)
# ==========================================================
PORTA_SERIAL = 'COM4'
VELOCIDADE = 9600

# Limiares de aceitação dos modelos
LIMIAR_CONFIANCA_GESTO = 0.65
LIMIAR_CONFIANCA_AMBIENTE = 0.50

# Mapeamento de classes do modelo para os caracteres que o Arduino entende
MAPEAMENTO_COMANDOS = {
    "frente": "f",
    "tras": "b",
    "esquerda": "l",
    "direita": "r",
    "parar": "s"
}

# Compartilhamento seguro de memória entre Threads
lock = threading.Lock()
rodando = True

frame_processado_gestos = None
frame_processado_ambiente = None

# ==========================================================
# 1. THREAD: CONTROLE DE GESTOS + ENVIO SERIAL (CAM 1)
# ==========================================================
def thread_controle_gestos():
    global frame_processado_gestos, rodando

    # 1.1 Inicialização da Conexão Serial com o Arduino
    print(f"[THREAD GESTOS] Tentando conectar na porta {PORTA_SERIAL}...")
    try:
        arduino = serial.Serial(PORTA_SERIAL, VELOCIDADE, timeout=1)
        time.sleep(2)  # Tempo crucial para o Arduino reiniciar e estabilizar
        print(f"[THREAD GESTOS] Arduino conectado com sucesso na porta {PORTA_SERIAL}!")
    except serial.SerialException as e:
        print(f"[THREAD GESTOS] AVISO: Não foi possível abrir a porta {PORTA_SERIAL} ({e}).")
        print("[THREAD GESTOS] O sistema rodará em MODO SIMULAÇÃO de comandos.")
        arduino = None

    # 1.2 Carregamento do Classificador (.pkl)
    print("[THREAD GESTOS] Carregando classificador de gestos (.pkl)...")
    try:
        with open("modelo_gestos.pkl", "rb") as f:
            modelo_gestos = pickle.load(f)
    except Exception as e:
        print(f"[THREAD GESTOS] ERRO fatal ao carregar 'modelo_gestos.pkl': {e}")
        rodando = False
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Mantendo o índice 1 conforme o seu último teste funcional
    cap_gestos = cv2.VideoCapture(1) 
    ultimo_comando_enviado = ""

    while rodando and cap_gestos.isOpened():
        sucesso, frame = cap_gestos.read()
        if not sucesso:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        gesto_atual = "NENHUMA MAO"
        comando_char = "s"  # Por padrão, se não há mão, o robô deve PARAR (Segurança)
        confianca_atual = 0.0

        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

            pulso_x, pulso_y = mao.landmark[0].x, mao.landmark[0].y
            vetor_caracteristicas = []
            for lm in mao.landmark:
                vetor_caracteristicas.append(lm.x - pulso_x)
                vetor_caracteristicas.append(lm.y - pulso_y)

            # Predição do Gesto Real
            probabilidades = modelo_gestos.predict_proba([vetor_caracteristicas])[0]
            confianca_atual = max(probabilidades)
            posicao_classe = probabilidades.argmax()
            classe_chutada = modelo_gestos.classes_[posicao_classe]

            if confianca_atual >= LIMIAR_CONFIANCA_GESTO:
                gesto_atual = classe_chutada.upper()
                comando_char = MAPEAMENTO_COMANDOS.get(classe_chutada, "s")
            else:
                gesto_atual = "DESCONHECIDO (ABAIXO DO LIMIAR)"
                comando_char = "s"

        # 1.3 Lógica Inteligente de Envio Serial (Anti-Spam)
        if comando_char != ultimo_comando_enviado:
            if arduino and arduino.is_open:
                arduino.write(comando_char.encode())
                print(f"[SERIAL] -> Enviado: '{comando_char}' | Motivo: {gesto_atual} ({confianca_atual:.2%})")
            else:
                print(f"[SIMULAÇÃO SERIAL] -> Enviar uma única vez: '{comando_char}' | Motivo: {gesto_atual}")
            ultimo_comando_enviado = comando_char

        # HUD Esquerdo (Gestos + Status da Serial)
        cor_hud = (0, 255, 0) if comando_char != "s" else (0, 0, 255)
        status_conexao = "CONECTADO" if arduino else "SIMULADO (DESCONECTADO)"
        
        cv2.putText(frame, "CAM 1 - CONTROLE POR GESTOS", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gesto: {gesto_atual}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_hud, 2)
        cv2.putText(frame, f"Certeza: {confianca_atual:.2%}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"CMD Serial: '{comando_char}'", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_hud, 2)
        cv2.putText(frame, f"Arduino: {status_conexao}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        with lock:
            frame_processado_gestos = frame.copy()

        time.sleep(0.01)

    # 1.4 Fechamento Seguro (Para o robô imediatamente se o script fechar)
    if arduino and arduino.is_open:
        arduino.write(b's')
        arduino.close()
        print("[THREAD GESTOS] Comando de parada enviado e conexão Serial encerrada.")
        
    cap_gestos.release()
    hands.close()

# ==========================================================
# 2. THREAD: RECONHECIMENTO DE AMBIENTES (CAM 0)
# ==========================================================
def thread_controle_ambiente():
    global frame_processado_ambiente, rodando

    caminho_modelo_amb = Path("yolo11m_salas.pt")
    if not caminho_modelo_amb.exists():
        print(f"[THREAD AMBIENTE] ERRO: Arquivo '{caminho_modelo_amb}' não encontrado!")
        rodando = False
        return

    print("[THREAD AMBIENTE] Carregando rede neural YOLOv11...")
    try:
        modelo_ambiente = YOLO(str(caminho_modelo_amb))
        print("[THREAD AMBIENTE] YOLOv11 pronto!")
    except Exception as e:
        print(f"[THREAD AMBIENTE] ERRO ao carregar YOLO: {e}")
        rodando = False
        return

    # Mantendo o índice 0 conforme o seu último teste funcional
    cap_ambiente = cv2.VideoCapture(0)

    while rodando:
        if not cap_ambiente.isOpened():
            img_erro = np.zeros((480, 640, 3), dtype=np.uint8) + 80
            cv2.putText(img_erro, "AGUARDANDO CONEXAO DA CAM 0 USB...", (40, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            with lock:
                frame_processado_ambiente = img_erro
            time.sleep(1)
            cap_ambiente = cv2.VideoCapture(0)
            continue

        sucesso, frame = cap_ambiente.read()
        if not sucesso:
            time.sleep(0.01)
            continue

        # Predição do Ambiente pelo YOLO
        resultados = modelo_ambiente.predict(frame, imgsz=224, verbose=False)

        ambiente_detectado = "BUSCANDO..."
        confianca_amb = 0.0

        if resultados and resultados[0].probs is not None:
            posicao_classe = resultados[0].probs.top1
            confianca_amb = resultados[0].probs.top1conf.item()
            ambiente_detectado = resultados[0].names[posicao_classe].upper()

        if confianca_amb >= LIMIAR_CONFIANCA_AMBIENTE:
            status_final = ambiente_detectado
            cor_hud_amb = (255, 191, 0)
        else:
            status_final = "IDENTIFICANDO..."
            cor_hud_amb = (0, 0, 255)

        # HUD Direito (Ambiente)
        cv2.putText(frame, "CAM 0 - RECONHECIMENTO DE SALA", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Sala: {status_final}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_hud_amb, 2)
        cv2.putText(frame, f"Certeza: {confianca_amb:.2%}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with lock:
            frame_processado_ambiente = frame.copy()

        time.sleep(0.01)

    cap_ambiente.release()

# ==========================================================
# 3. THREAD PRINCIPAL (CONCATENAÇÃO E MONITORAMENTO)
# ==========================================================
print("=" * 60)
print("INICIANDO SISTEMA INTEGRADO DE TELEOPERAÇÃO - TCC")
print("=" * 60)

t_gestos = threading.Thread(target=thread_controle_gestos, daemon=True)
t_ambiente = threading.Thread(target=thread_controle_ambiente, daemon=True)

t_gestos.start()
t_ambiente.start()

try:
    while rodando:
        img_esquerda = None
        img_direita = None

        with lock:
            if frame_processado_gestos is not None:
                img_esquerda = frame_processado_gestos.copy()
            if frame_processado_ambiente is not None:
                img_direita = frame_processado_ambiente.copy()

        if img_esquerda is not None and img_direita is not None:
            altura_padrao = 480
            largura_padrao = 640
            
            img_esq_res = cv2.resize(img_esquerda, (largura_padrao, altura_padrao))
            img_dir_res = cv2.resize(img_direita, (largura_padrao, altura_padrao))

            # Concatenação lado a lado (Split-Screen)
            tela_dividida = np.hstack((img_esq_res, img_dir_res))
            cv2.line(tela_dividida, (largura_padrao, 0), (largura_padrao, altura_padrao), (0, 0, 0), 5)

            cv2.imshow("SISTEMA CO-PILOTO UNIFICADO - TCC", tela_dividida)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rodando = False
            break

except KeyboardInterrupt:
    print("\nEncerrando via terminal...")
    rodando = False

rodando = False
t_gestos.join()
t_ambiente.join()

cv2.destroyAllWindows()
print("\n[FINALIZADO] Sistema desconectado com sucesso.")