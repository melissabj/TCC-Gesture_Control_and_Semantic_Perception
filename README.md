# TCC-Gesture_Control_and_Semantic_Perception
## Gesture Control & Semantic Scene Perception for Mobile Robotics

Este repositório contém o código-fonte e os modelos treinados do sistema de Dupla Percepção Visual desenvolvido para o controle por gestos e classificação semântica de ambientes em um protótipo de robô móvel indoor.

# 📌 Sumário

- Sobre o Projeto

- Estrutura do Repositório

- Tecnologias Utilizadas

- Instalação

- Como Executar

- Mapeamento de Gestos e Comandos

- Resultados Obtidos

- Autora

## 📖 Sobre o Projeto

O sistema opera de forma integrada processando duas tarefas em tempo real a partir do fluxo da webcam:

    Controle Gestual (Teleoperação): Captura a mão via MediaPipe Hands, extrai os 21 pontos-chave (landmarks) e classifica a direção do movimento utilizando o modelo Random Forest (Modelo_gestos.pkl).

    Percepção Semântica do Ambiente: Identifica a categoria do espaço físico (Laboratório, Sala de Aula ou Sala de Professores) utilizando a rede neural YOLO11m ajustada (yolo11m_salas.pt).

Os comandos validados são transmitidos serialmente para um microcontrolador Arduino para acionamento dos motores.

## 📂 Estrutura do Repositório
```bash

TCC-Gesture_Control_and_Semantic_Perception/
└── code+models/
    ├── Modelo_gestos.pkl               # Modelo Random Forest treinado para gestos
    ├── yolo11m_salas.pt                # Pesos do YOLO11m para classificação de cenas
    ├── sistema_unificado.py            # Script principal de execução e controle
    └── sistema_unificado_telemetria.py  # Script de execução com coleta de métricas (FPS, latência)
```

## 🛠 Tecnologias Utilizadas

| Categoria | Biblioteca / Tecnologia | Função no Projeto |
| --------- | ----------------------- | ----------------- |
| Linguagem | Python 3.10+ | Desenvolvimento dos scripts unificados |
| Visão Computacional | OpenCV | Processamento e exibição de frames da webcam |
| | MediaPipe | Detecção e extração dos 21 landmarks da mão |
| Deep Learning & ML | Ultralytics (ultralytics) | Execução da inferência do YOLO11m (.pt) |
| | Scikit-Learn (scikit-learn) | Inferência do classificador Random Forest (.pkl) |
| Comunicação | PySerial (pyserial) | Comunicação serial USB entre Python e Arduino |

## 🚀 Instalação

Clone o repositório:
  
```bash
    git clone https://github.com/melissabj/TCC-Gesture_Control_and_Semantic_Perception.git
    cd TCC-Gesture_Control_and_Semantic_Perception
```
Instale as dependências diretamente via pip:
```bash
    pip install opencv-python mediapipe ultralytics pyserial scikit-learn
```

## 🎮 Como Executar

### 1. Modo Padrão (Controle e Percepção em Tempo Real)

Executa a interface visual da câmera, reconhece os gestos e a cena simultaneamente e envia os comandos via porta serial:
```Bash
python code+models/sistema_unificado.py
```

### 2. Modo com Telemetria e Coleta de Métricas

Executa o mesmo pipeline de visão, adicionando a medição em tempo real e o registro de dados de desempenho (Frames Por Segundo - FPS, tempo de inferência e latência do sistema):
```Bash
python code+models/sistema_unificado_telemetria.py
```

Nota: Caso esteja executando sem o Arduino conectado, certifique-se de tratar a exceção da porta serial ou ajustar a variável de conexão no script para testes locais.

## 🖐 Mapeamento de Gestos e Comandos

| Gesto Detectado | Comando Serial | Ação no Robô |
| --------------- | -------------- | ------------ |
| Mão Aberta | F | Avanço (Forward) |
| Mão Aberta | S | Processamento e exibição de frames da webcam |
| Indicador p/ Esquerda | L | |
| Dedão p/ Esquerda | | Giro à Esquerda (Left) |
| Indicador p/ Direita | R | |
| Dedão p/ Direita | | Giro à Direita (Right) |
| Sinal de Pare (Palma) | S | Parada (Stop) |

## 📊 Resultados Obtidos

- Acurácia na Classificação de Cenas: 91,69%

- Taxa Média de Processamento: 132,4 FPS

- Latência Média de Inferência: 7,6 ms

## 👩‍💻 Autora

Melissa Batista Junqueira
Estudante de Tecnologia em Análise e Desenvolvimento de Sistemas

GitHub: @melissabj
