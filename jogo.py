import pygame
from pygame.locals import *
from sys import exit
import os
from random import randrange, choice
import cv2
import mediapipe as mp
import threading
import numpy as np
import sys

# Adicione no início do código:
if getattr(sys, 'frozen', False):
    os.environ['MEDIAPIPE_MODELS_PATH'] = os.path.join(sys._MEIPASS, 'mediapipe/modules')
else:
    diretorio_principal = os.path.dirname(__file__)

pygame.init()
pygame.mixer.init()

# Configurações do MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
jump_event = threading.Event()
JUMP_THRESHOLD = -0.03

# Configurações de escala
BASE_WIDTH, BASE_HEIGHT = 640, 480
desktop_info = pygame.display.Info()
SCREEN_WIDTH = desktop_info.current_w
SCREEN_HEIGHT = desktop_info.current_h
scale = min(SCREEN_WIDTH / BASE_WIDTH, SCREEN_HEIGHT / BASE_HEIGHT)
scaled_width = int(BASE_WIDTH * scale)
scaled_height = int(BASE_HEIGHT * scale)

# Variáveis globais para compartilhamento entre threads
global_frame = None
frame_lock = threading.Lock()

# Inicialização da janela
tela = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
base_surface = pygame.Surface((BASE_WIDTH, BASE_HEIGHT))
pygame.display.set_caption('Dino')

# Configurações do jogo
diretorio_principal = os.path.dirname(__file__)
diretorio_imagens = os.path.join(diretorio_principal, 'imagens')
diretorio_sons = os.path.join(diretorio_principal, 'sons')

sprite_sheet = pygame.image.load(os.path.join(diretorio_imagens, 'spritesheet.png')).convert_alpha()


def webcam_jump_detection():
    global global_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_y = None
    smoothed_y = None
    alpha = 0.7

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True

            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                current_y = nose.y

                if smoothed_y is None:
                    smoothed_y = current_y
                else:
                    smoothed_y = alpha * current_y + (1 - alpha) * smoothed_y

                if prev_y is not None:
                    delta_y = smoothed_y - prev_y
                    if delta_y < JUMP_THRESHOLD:
                        jump_event.set()

                prev_y = smoothed_y

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Processamento do frame para exibição no Pygame
            image = cv2.flip(image, 1)
            resized_image = cv2.resize(image, (200, 150))

            with frame_lock:
                global_frame = resized_image.copy()

    cap.release()


# Iniciar thread da webcam
thread = threading.Thread(target=webcam_jump_detection, daemon=True)
thread.start()

# Sons do jogo
som_colisao = pygame.mixer.Sound(os.path.join(diretorio_sons, 'death_sound.wav'))
som_colisao.set_volume(1)
som_pontuacao = pygame.mixer.Sound(os.path.join(diretorio_sons, 'score_sound.wav'))
som_pontuacao.set_volume(1)

# Variáveis do jogo
colidiu = False
escolha_obstaculo = choice([0, 1])
pontos = 0
velocidade_jogo = 10

def exibe_mensagem(msg, tamanho, cor):
    fonte = pygame.font.SysFont('comicsansms', tamanho, True, False)
    return fonte.render(msg, True, cor)

def reiniciar_jogo():
    global pontos, velocidade_jogo, colidiu, escolha_obstaculo
    pontos = 0
    velocidade_jogo = 10
    colidiu = False
    dino.rect.y = BASE_HEIGHT - 64 - 96//2
    dino.pulo = False
    dino_voador.rect.x = BASE_WIDTH
    cacto.rect.x = BASE_WIDTH
    escolha_obstaculo = choice([0, 1])

class Dino(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.som_pulo = pygame.mixer.Sound(os.path.join(diretorio_sons, 'jump_sound.wav'))
        self.som_pulo.set_volume(1)
        self.imagens_dinossauro = []
        for i in range(2):
            frame = sprite_sheet.subsurface((i*32, 0), (32, 32))
            frame = pygame.transform.scale(frame, (32*3, 32*3))
            self.imagens_dinossauro.append(frame)
        self.index_lista = 0
        self.image = self.imagens_dinossauro[0]
        self.pos_y_inicial = BASE_HEIGHT - 75 - 96//2
        self.rect = self.image.get_rect(center=(175, BASE_HEIGHT - 64))
        self.mask = pygame.mask.from_surface(self.image)
        self.pulo = False

    def pular(self):
        self.pulo = True
        self.som_pulo.play()

    def update(self):
        if self.pulo:
            if self.rect.y <= 160:
                self.pulo = False
            self.rect.y -= 30
        else:
            if self.rect.y < self.pos_y_inicial:
                self.rect.y += 20
            else:
                self.rect.y = self.pos_y_inicial
        self.index_lista += 0.25
        if self.index_lista >= len(self.imagens_dinossauro):
            self.index_lista = 0
        self.image = self.imagens_dinossauro[int(self.index_lista)]

class Nuvens(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = sprite_sheet.subsurface((11*32, 0), (32, 32))
        self.image = pygame.transform.scale(self.image, (32*3, 32*3))
        self.rect = self.image.get_rect()
        self.rect.y = randrange(50, 200, 50)
        self.rect.x = BASE_WIDTH - randrange(30, 300, 90)

    def update(self):
        if self.rect.topright[0] < 0:
            self.rect.x = BASE_WIDTH
            self.rect.y = randrange(50, 200, 50)
        self.rect.x -= velocidade_jogo


class Chao(pygame.sprite.Sprite):
    def __init__(self, pos_x):
        pygame.sprite.Sprite.__init__(self)
        self.image = sprite_sheet.subsurface((10 * 32, 0), (32, 32))
        self.image = pygame.transform.scale(self.image, (32 * 2, 32 * 2))  # 64x64
        self.rect = self.image.get_rect()
        self.rect.y = BASE_HEIGHT - 64
        self.rect.x = pos_x * 64

    def update(self):
        # Apenas movimento, sem reposicionamento
        self.rect.x -= velocidade_jogo


class Cacto(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.escolha = escolha_obstaculo
        self.escolher_frame()  # Agora cria o rect com posição correta
        self.mask = pygame.mask.from_surface(self.image)

    def escolher_frame(self):
        # Lista de frames baseada na pontuação
        frames_disponiveis = [5, 6]
        if pontos >= 500:
            frames_disponiveis += [7, 8, 9]  # Adiciona novos frames

        frame = choice(frames_disponiveis)
        self.image = sprite_sheet.subsurface((frame * 32, 0), (32, 32))
        self.image = pygame.transform.scale(self.image, (32 * 3, 32 * 3))

        # Ajusta posição Y para frames específicos
        self.rect = self.image.get_rect()
        if frame in [6, 7, 8, 9]:  # Frames altos
            self.rect.centery = BASE_HEIGHT - 50 - 5  # 15px mais alto
        else:
            self.rect.centery = BASE_HEIGHT - 50  # Posição normal

        self.rect.x = BASE_WIDTH  # Posiciona inicialmente fora da tela

    def update(self):
        if self.escolha == 0:
            if self.rect.topright[0] < 0:
                self.rect.x = BASE_WIDTH
            self.rect.x -= velocidade_jogo

class DinoVoador(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.imagens_dinossauro = []
        for i in range(3, 5):
            img = sprite_sheet.subsurface((i*32, 0), (32, 32))
            img = pygame.transform.scale(img, (32*4, 32*4))
            self.imagens_dinossauro.append(img)
        self.index_lista = 0
        self.image = self.imagens_dinossauro[self.index_lista]
        self.mask = pygame.mask.from_surface(self.image)
        self.escolha = escolha_obstaculo
        self.rect = self.image.get_rect()
        self.rect.center = (BASE_WIDTH, 300)
        self.rect.x = BASE_WIDTH

    def update(self):
        if self.escolha == 1:
            if self.rect.topright[0] < 0:
                self.rect.x = BASE_WIDTH
            self.rect.x -= velocidade_jogo

            self.index_lista += 0.10
            if self.index_lista >= len(self.imagens_dinossauro):
                self.index_lista = 0
            self.image = self.imagens_dinossauro[int(self.index_lista)]

# Inicialização dos objetos do jogo
todas_as_sprites = pygame.sprite.Group()
dino = Dino()
todas_as_sprites.add(dino)

for _ in range(4):
    todas_as_sprites.add(Nuvens())

quantidade_de_chaos = (BASE_WIDTH // 64) + 2  # Cobre a tela + margem
for i in range(quantidade_de_chaos):
    todas_as_sprites.add(Chao(i))

cacto = Cacto()
dino_voador = DinoVoador()
grupo_obstaculos = pygame.sprite.Group(cacto, dino_voador)
todas_as_sprites.add(cacto, dino_voador)

imagem_fundo = pygame.image.load('imagens/bg_chickenRun.png').convert()
imagem_fundo = pygame.transform.scale(imagem_fundo, (BASE_WIDTH, BASE_HEIGHT))

relogio = pygame.time.Clock()

# ... (código anterior permanece igual até o loop principal)

while True:
    relogio.tick(30)
    base_surface.blit(imagem_fundo, (0, 0))

    frame_surface = None
    with frame_lock:
        if global_frame is not None:
            frame_surface = pygame.image.frombuffer(
                global_frame.tobytes(),
                (global_frame.shape[1], global_frame.shape[0]),
                'RGB'
            )

    if frame_surface:
        base_surface.blit(frame_surface, (10, 10))

    # Eventos
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:  # Tecla ESC para sair
                pygame.quit()
                exit()
            if event.key == K_r and colidiu:
                reiniciar_jogo()

    # Lógica do jogo
    if not colidiu and jump_event.is_set():
        if dino.rect.y == dino.pos_y_inicial:
            dino.pular()
        jump_event.clear()

    colisoes = pygame.sprite.spritecollide(dino, grupo_obstaculos, False, pygame.sprite.collide_mask)

    # Atualização e desenho (só atualiza se não colidiu)
    if not colidiu:
        todas_as_sprites.update()

        # Reposicionamento do chão após todas as atualizações
        chao_sprites = [sprite for sprite in todas_as_sprites if isinstance(sprite, Chao)]
        for chao in chao_sprites:
            if chao.rect.right < 0:
                # Encontra o sprite mais à direita
                max_x = max((s.rect.x for s in chao_sprites), default=-64)
                chao.rect.x = max_x + 64
    else:  # Congela a posição dos obstáculos
        cacto.rect.x += 0
        dino_voador.rect.x += 0

    todas_as_sprites.draw(base_surface)

    # Lógica dos obstáculos
    # Dentro do loop principal, na seção "Lógica dos obstáculos":
    # Dentro do loop principal, na seção "Lógica dos obstáculos":
    if cacto.rect.topright[0] <= 0 or dino_voador.rect.topright[0] <= 0:
        escolha_obstaculo = choice([0, 1])
        cacto.rect.x = dino_voador.rect.x = BASE_WIDTH
        cacto.escolha = dino_voador.escolha = escolha_obstaculo

        # Se for cacto, escolhe novo frame e reposiciona
        if escolha_obstaculo == 0:
            cacto.escolher_frame()  # Isso já ajusta a posição Y automaticamente

    # Game Over
    if colisoes and not colidiu:
        som_colisao.play()
        colidiu = True

    # Interface
    if colidiu:
        game_over = exibe_mensagem('GAME OVER', 40, (0, 0, 0))
        restart = exibe_mensagem('Pressione R para reiniciar', 20, (0, 0, 0))
        base_surface.blit(game_over, (BASE_WIDTH // 2 - game_over.get_width() // 2, BASE_HEIGHT // 2 - 50))
        base_surface.blit(restart, (BASE_WIDTH // 2 - restart.get_width() // 2, BASE_HEIGHT // 2))
    else:
        pontos += 1
        texto_pontos = exibe_mensagem(str(pontos), 40, (0, 0, 0))
        base_surface.blit(texto_pontos, (520, 30))

        if pontos % 100 == 0:
            som_pontuacao.play()
            velocidade_jogo = min(velocidade_jogo + 1, 23)

    # Atualização da tela
    scaled_surface = pygame.transform.smoothscale(base_surface, (scaled_width, scaled_height))
    tela.blit(scaled_surface, ((SCREEN_WIDTH - scaled_width) // 2, (SCREEN_HEIGHT - scaled_height) // 2))

    pygame.display.flip()