import cv2
import csv
import pickle
import time
import serial
import threading
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==========================================================
# CONFIGURAÇÕES DA SERIAL, ARQUIVOS E LIMIARES
# ==========================================================
PORTA_SERIAL = 'COM4'
VELOCIDADE = 9600

LIMIAR_CONFIANCA_GESTO = 0.45
LIMIAR_CONFIANCA_AMBIENTE = 0.50

MAPEAMENTO_COMANDOS = {
    "frente": "f",
    "tras": "b",
    "esquerda": "l",
    "direita": "r",
    "parar": "s"
}

# Nomes únicos baseados na data e hora da execução
timestamp_inicio = datetime.now().strftime("%Y%m%d_%H%M%S")
NOME_VIDEO = f"teste_tcc_{timestamp_inicio}.mp4"
NOME_CSV = f"telemetria_tcc_{timestamp_inicio}.csv"

# Estrutura global de telemetria e sincronização
lock = threading.Lock()
rodando = True

frame_processado_gestos = None
frame_processado_ambiente = None

telemetria = {
    "gesto": "NENHUMA MAO",
    "conf_gesto": 0.0,
    "cmd_serial": "s",
    "sala": "BUSCANDO...",
    "conf_sala": 0.0
}

# ==========================================================
# 1. THREAD: CONTROLE DE GESTOS + SERIAL (CAM 1)
# ==========================================================
def thread_controle_gestos():
    global frame_processado_gestos, rodando, telemetria

    try:
        arduino = serial.Serial(PORTA_SERIAL, VELOCIDADE, timeout=1)
        time.sleep(2)
        print(f"[THREAD GESTOS] Arduino conectado na porta {PORTA_SERIAL}!")
    except Exception:
        print("[THREAD GESTOS] Modo SIMULAÇÃO ativado (Arduino não encontrado).")
        arduino = None

    try:
        with open("modelo_gestos.pkl", "rb") as f:
            modelo_gestos = pickle.load(f)
    except Exception as e:
        print(f"[THREAD GESTOS] ERRO ao carregar modelo: {e}")
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
        comando_char = "s"
        confianca_atual = 0.0

        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

            pulso_x, pulso_y = mao.landmark[0].x, mao.landmark[0].y
            vetor_caracteristicas = []
            for lm in mao.landmark:
                vetor_caracteristicas.append(lm.x - pulso_x)
                vetor_caracteristicas.append(lm.y - pulso_y)

            probabilidades = modelo_gestos.predict_proba([vetor_caracteristicas])[0]
            confianca_atual = float(max(probabilidades))
            posicao_classe = probabilidades.argmax()
            classe_chutada = modelo_gestos.classes_[posicao_classe]

            if confianca_atual >= LIMIAR_CONFIANCA_GESTO:
                gesto_atual = classe_chutada.upper()
                comando_char = MAPEAMENTO_COMANDOS.get(classe_chutada, "s")
            else:
                gesto_atual = "DESCONHECIDO"
                comando_char = "s"

        if comando_char != ultimo_comando_enviado:
            if arduino and arduino.is_open:
                arduino.write(comando_char.encode())
            ultimo_comando_enviado = comando_char

        cor_hud = (0, 255, 0) if comando_char != "s" else (0, 0, 255)
        cv2.putText(frame, "CAM 1 - GESTOS", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gesto: {gesto_atual}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_hud, 2)
        cv2.putText(frame, f"Certeza: {confianca_atual:.2%}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with lock:
            frame_processado_gestos = frame.copy()
            telemetria["gesto"] = gesto_atual
            telemetria["conf_gesto"] = confianca_atual
            telemetria["cmd_serial"] = comando_char

        time.sleep(0.01)

    if arduino and arduino.is_open:
        arduino.write(b's')
        arduino.close()
    cap_gestos.release()
    hands.close()

# ==========================================================
# 2. THREAD: RECONHECIMENTO DE AMBIENTES (CAM 0)
# ==========================================================
def thread_controle_ambiente():
    global frame_processado_ambiente, rodando, telemetria

    caminho_modelo_amb = Path("yolo11m_salas.pt")
    if not caminho_modelo_amb.exists():
        rodando = False
        return

    try:
        modelo_ambiente = YOLO(str(caminho_modelo_amb))
    except Exception:
        rodando = False
        return

    cap_ambiente = cv2.VideoCapture(0)

    while rodando:
        if not cap_ambiente.isOpened():
            img_erro = np.zeros((480, 640, 3), dtype=np.uint8) + 80
            cv2.putText(img_erro, "AGUARDANDO CAM 0...", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            with lock:
                frame_processado_ambiente = img_erro
            time.sleep(1)
            cap_ambiente = cv2.VideoCapture(0)
            continue

        sucesso, frame = cap_ambiente.read()
        if not sucesso:
            time.sleep(0.01)
            continue

        resultados = modelo_ambiente.predict(frame, imgsz=224, verbose=False)

        ambiente_detectado = "BUSCANDO..."
        confianca_amb = 0.0

        if resultados and resultados[0].probs is not None:
            posicao_classe = resultados[0].probs.top1
            confianca_amb = float(resultados[0].probs.top1conf.item())
            ambiente_detectado = resultados[0].names[posicao_classe].upper()

        if confianca_amb >= LIMIAR_CONFIANCA_AMBIENTE:
            status_final = ambiente_detectado
            cor_hud_amb = (255, 191, 0)
        else:
            status_final = "IDENTIFICANDO..."
            cor_hud_amb = (0, 0, 255)

        cv2.putText(frame, "CAM 0 - RECONHECIMENTO SALA", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Sala: {status_final}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_hud_amb, 2)
        cv2.putText(frame, f"Certeza: {confianca_amb:.2%}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with lock:
            frame_processado_ambiente = frame.copy()
            telemetria["sala"] = status_final
            telemetria["conf_sala"] = confianca_amb

        time.sleep(0.01)

    cap_ambiente.release()

# ==========================================================
# 3. THREAD PRINCIPAL (Gravação MP4, Telemetria CSV & Display)
# ==========================================================
print("=" * 60)
print("INICIANDO SISTEMA COM GRAVAÇÃO E LOG DE TELEMETRIA")
print(f"Vídeo de Saída: {NOME_VIDEO}")
print(f"Relatório CSV:  {NOME_CSV}")
print("=" * 60)

# Configuração do Gravador MP4 (Lado a lado = 1280x480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
gravador_video = cv2.VideoWriter(NOME_VIDEO, fourcc, 20.0, (1280, 480))

# Configuração do Log CSV
arquivo_csv = open(NOME_CSV, mode='w', newline='', encoding='utf-8')
escritor_csv = csv.writer(arquivo_csv)
escritor_csv.writerow([
    "tempo_decorrido_s",
    "fps_sistema",
    "gesto_detectado",
    "confianca_gesto",
    "comando_serial",
    "sala_detectada",
    "confianca_sala"
])

t_gestos = threading.Thread(target=thread_controle_gestos, daemon=True)
t_ambiente = threading.Thread(target=thread_controle_ambiente, daemon=True)

t_gestos.start()
t_ambiente.start()

tempo_inicio_script = time.time()
tempo_frame_anterior = time.time()

try:
    while rodando:
        img_esquerda = None
        img_direita = None

        with lock:
            if frame_processado_gestos is not None:
                img_esquerda = frame_processado_gestos.copy()
            if frame_processado_ambiente is not None:
                img_direita = frame_processado_ambiente.copy()
            
            telemetria_atual = telemetria.copy()

        if img_esquerda is not None and img_direita is not None:
            # Cálculo do FPS do pipeline visual
            tempo_atual = time.time()
            fps = 1.0 / (tempo_atual - tempo_frame_anterior) if (tempo_atual - tempo_frame_anterior) > 0 else 0.0
            tempo_frame_anterior = tempo_atual
            tempo_decorrido = tempo_atual - tempo_inicio_script

            img_esq_res = cv2.resize(img_esquerda, (640, 480))
            img_dir_res = cv2.resize(img_direita, (640, 480))

            tela_dividida = np.hstack((img_esq_res, img_dir_res))
            cv2.line(tela_dividida, (640, 0), (640, 480), (0, 0, 0), 5)

            # Exibe FPS na tela unificada
            cv2.putText(tela_dividida, f"FPS: {fps:.1f}", (1160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 1. Grava frame no vídeo .mp4
            gravador_video.write(tela_dividida)

            # 2. Grava linha no arquivo .csv
            escritor_csv.writerow([
                f"{tempo_decorrido:.2f}",
                f"{fps:.2f}",
                telemetria_atual["gesto"],
                f"{telemetria_atual['conf_gesto']:.4f}",
                telemetria_atual["cmd_serial"],
                telemetria_atual["sala"],
                f"{telemetria_atual['conf_sala']:.4f}"
            ])

            cv2.imshow("TELEMETRIA E GRAVACAO EM TEMPO REAL - TCC", tela_dividida)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rodando = False
            break

except KeyboardInterrupt:
    rodando = False

# Encerramento seguro e liberação de buffers de mídia
rodando = False
t_gestos.join()
t_ambiente.join()

gravador_video.release()
arquivo_csv.close()
cv2.destroyAllWindows()

print(f"\n[SUCESSO] Gravação '{NOME_VIDEO}' e log '{NOME_CSV}' salvos com sucesso!")