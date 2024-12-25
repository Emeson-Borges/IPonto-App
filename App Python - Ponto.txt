import kivy
kivy.require("2.1.0")

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle


from cryptography.fernet import Fernet
from deepface import DeepFace
import cv2
import face_recognition
import numpy as np
import sqlite3
import json
import os
import math
import hashlib

from datetime import datetime
import threading
import requests  # para exemplo de sincronizar (pip install requests)

# -----------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# -----------------------------------------------------
Window.size = (400, 720)  # Simular tela maior
DB_PATH = "faces.db"
MODEL_TYPE = "hog"        # ou "cnn"
EAR_THRESHOLD = 0.22
CONSEC_FRAMES_CLOSED = 1
CONSEC_FRAMES_OPEN = 1
FACE_RECOG_TOLERANCE = 0.4  # Ajustável

# -----------------------------------------------------
# BANCO DE DADOS
# -----------------------------------------------------
def create_db_if_not_exists():
    """
    Cria as tabelas necessárias no banco de dados, incluindo a coluna única para o hash.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            encoding_hash TEXT UNIQUE NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            date_time TEXT NOT NULL,
            log_type TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()



def update_db_schema():
    """
    Atualiza o esquema do banco de dados, adicionando a coluna log_type se necessário.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Adicionar a coluna log_type, se ela não existir
        c.execute("ALTER TABLE face_logs ADD COLUMN log_type TEXT")
    except sqlite3.OperationalError:
        # A coluna já existe
        pass
    conn.commit()
    conn.close()

def face_already_exists(encoding):
    """
    Verifica se o rosto já foi cadastrado no banco de dados com base no hash do encoding.
    Permite novos registros se os encodings forem únicos.
    """
    normalized_encoding = np.array(encoding, dtype=np.float32) / np.linalg.norm(encoding)
    encoding_hash = hashlib.sha256(json.dumps(normalized_encoding.tolist()).encode()).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM face_encodings WHERE encoding_hash = ?', (encoding_hash,))
    count = c.fetchone()[0]
    conn.close()

    return count > 0



def user_already_exists(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM face_encodings WHERE name=?', (name,))
    count = c.fetchone()[0]
    conn.close()
    return (count > 0)

def save_face_encoding(name, encoding):
    """
    Salva o encoding do rosto no banco de dados, verificando se já existe.
    """
    # Normalizar o encoding
    normalized_encoding = np.array(encoding, dtype=np.float32) / np.linalg.norm(encoding)

    # Gerar o hash do encoding normalizado
    encoding_json = json.dumps(normalized_encoding.tolist())
    encoding_hash = hashlib.sha256(encoding_json.encode()).hexdigest()

    # Conexão com o banco
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Verificar se o hash já existe
    c.execute('SELECT name, encoding FROM face_encodings WHERE encoding_hash = ?', (encoding_hash,))
    result = c.fetchone()
    if result:
        raise ValueError(f"O rosto já está cadastrado como '{result[0]}'.")

    # Verificar duplicidade com base na similaridade facial
    c.execute('SELECT name, encoding FROM face_encodings')
    rows = c.fetchall()
    for existing_name, existing_enc_json in rows:
        existing_encoding = np.array(json.loads(existing_enc_json), dtype=np.float32)
        distance = face_recognition.face_distance([existing_encoding], normalized_encoding)[0]
        if distance < FACE_RECOG_TOLERANCE:  # Apenas bloquear se muito similar
            raise ValueError(f"O rosto já está cadastrado como '{existing_name}' (similaridade detectada).")

    # Inserir o novo rosto no banco
    c.execute(
        'INSERT INTO face_encodings (name, encoding, encoding_hash) VALUES (?, ?, ?)',
        (name, encoding_json, encoding_hash)
    )
    conn.commit()
    conn.close()




def remove_duplicate_encodings():
    """
    Remove encodings duplicados do banco de dados, deixando apenas o primeiro registro.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        DELETE FROM face_encodings
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM face_encodings
            GROUP BY encoding_hash
        )
    ''')
    conn.commit()
    conn.close()


def load_face_encodings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name, encoding FROM face_encodings')
    rows = c.fetchall()
    conn.close()

    known_faces = {}
    for (name, enc_json) in rows:
        arr = np.array(json.loads(enc_json), dtype=np.float32)
        if name not in known_faces:
            known_faces[name] = []
        known_faces[name].append(arr)
    return known_faces

def save_log_punch(name, encoding):
    """
    Salva o log de batida com informação de entrada ou saída.
    """
    enc_json = json.dumps(encoding.tolist())
    now = datetime.now()
    now_date = now.strftime("%Y-%m-%d")
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Verifica quantas batidas já ocorreram no mesmo dia para este usuário
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM face_logs WHERE name = ? AND date_time LIKE ?', (name, now_date + "%"))
    log_count_today = c.fetchone()[0]
    conn.close()

    # Define o tipo com base na ordem
    log_type = "Entrada" if log_count_today % 2 == 0 else "Saída"

    # Salva no banco de dados
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO face_logs (name, encoding, date_time, log_type) VALUES (?, ?, ?, ?)',
              (name, enc_json, now_str, log_type))
    conn.commit()
    conn.close()

def ensure_encoding_hash_column():
    """
    Garante que a coluna 'encoding_hash' exista na tabela 'face_encodings'.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Verificar se a coluna 'encoding_hash' já existe
    c.execute("PRAGMA table_info(face_encodings)")
    columns = [row[1] for row in c.fetchall()]

    if 'encoding_hash' not in columns:
        # Adicionar a coluna se não existir
        c.execute("ALTER TABLE face_encodings ADD COLUMN encoding_hash TEXT UNIQUE")

    conn.commit()
    conn.close()



def update_existing_encodings_with_hash():
    """
    Atualiza todos os registros existentes na tabela 'face_encodings' para incluir o hash do encoding.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Busca registros que ainda não têm o hash calculado
    c.execute("SELECT id, encoding FROM face_encodings WHERE encoding_hash IS NULL")
    rows = c.fetchall()

    for row_id, encoding_json in rows:
        encoding = np.array(json.loads(encoding_json), dtype=np.float32)
        encoding_hash = hashlib.sha256(json.dumps(encoding.tolist()).encode()).hexdigest()
        c.execute("UPDATE face_encodings SET encoding_hash = ? WHERE id = ?", (encoding_hash, row_id))

    conn.commit()
    conn.close()



def load_all_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, date_time, encoding FROM face_logs ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return rows

key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_encoding(encoding):
    encoding_json = json.dumps(encoding.tolist())
    return cipher.encrypt(encoding_json.encode())

def decrypt_encoding(encrypted_encoding):
    return json.loads(cipher.decrypt(encrypted_encoding).decode())


# -----------------------------------------------------
# FUNÇÕES FACE RECOGNITION
# -----------------------------------------------------
def detect_main_face(frame_rgb):
    locs = face_recognition.face_locations(frame_rgb, model=MODEL_TYPE)
    if not locs:
        return None
    best_area = 0
    best_loc = None
    for (top, right, bottom, left) in locs:
        area = (bottom - top) * (right - left)
        if area > best_area:
            best_area = area
            best_loc = (top, right, bottom, left)
    return best_loc

def get_face_encoding(frame_bgr):
    """
    Obtém o encoding do rosto na imagem e garante consistência no processo.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    loc = detect_main_face(frame_rgb)
    if loc is None:
        return None, None
    encs = face_recognition.face_encodings(frame_rgb, known_face_locations=[loc], num_jitters=1)
    if len(encs) == 0:
        return None, None
    # Normalizar o encoding para consistência
    normalized_enc = np.array(encs[0], dtype=np.float32) / np.linalg.norm(encs[0])
    return normalized_enc, loc



def compare_with_db(unknown_enc, unknown_img):
    """
    Compara um rosto desconhecido com os registros no banco de dados.
    """
    known_faces = load_face_encodings()
    if not known_faces:
        return None  # Nenhum rosto registrado

    best_name = None
    best_dist = float('inf')

    for name, encodings_list in known_faces.items():
        for known_enc in encodings_list:
            # Comparação inicial com face_recognition
            dist = face_recognition.face_distance([known_enc], unknown_enc)[0]
            if dist < best_dist and dist < FACE_RECOG_TOLERANCE:
                best_name = name
                best_dist = dist

    return best_name




def is_duplicate_face(new_encoding):
    """
    Verifica se o rosto já existe no banco de dados com base no hash ou na similaridade facial.
    """
    new_encoding_hash = hashlib.sha256(json.dumps(new_encoding.tolist()).encode()).hexdigest()

    # Verifica no banco de dados
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT encoding, encoding_hash FROM face_encodings')
    rows = c.fetchall()
    conn.close()

    for encoding_json, encoding_hash in rows:
        existing_encoding = np.array(json.loads(encoding_json))

        # Comparação por hash
        if encoding_hash == new_encoding_hash:
            return True

        # Comparação por similaridade facial
        dist = face_recognition.face_distance([existing_encoding], new_encoding)[0]
        if dist < FACE_RECOG_TOLERANCE:
            return True

    return False




def is_real_face(frame, loc):
    """
    Verifica se o rosto é real analisando a textura da imagem.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = loc
    roi = gray[y:h, x:w]
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    return laplacian_var > 50  # Ajuste o limiar conforme necessário

def average_encodings(encodings):
    """
    Calcula a média de múltiplos encodings para criar um perfil robusto.
    """
    return np.mean(encodings, axis=0)

def verify_face(image1, image2):
    """
    Verifica se dois rostos correspondem usando DeepFace.
    """
    result = DeepFace.verify(image1, image2, model_name="Facenet")
    return result['verified']

# -----------------------------------------------------
# DETECÇÃO DE PISCADA
# -----------------------------------------------------
def face_landmarks_for_loc(frame_rgb, loc):
    lands = face_recognition.face_landmarks(frame_rgb, [loc])
    if len(lands) == 0:
        return None
    return lands[0]

def calc_ear(eye_points):
    if len(eye_points) != 6:
        return 1.0
    def euclidian(a, b):
        return math.dist(a, b)
    p1, p2, p3, p4, p5, p6 = eye_points
    dist2_6 = euclidian(p2, p6)
    dist3_5 = euclidian(p3, p5)
    dist1_4 = euclidian(p1, p4)
    ear = (dist2_6 + dist3_5) / (2.0 * dist1_4)
    return ear

def detect_blink(landmarks, blink_state):
    """
    Detecta piscadas com base na razão de aspecto do olho (EAR).
    """
    left_eye = landmarks.get('left_eye')
    right_eye = landmarks.get('right_eye')
    if not left_eye or not right_eye:
        return False

    left_ear = calc_ear(left_eye)
    right_ear = calc_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    eyes_closed = avg_ear < EAR_THRESHOLD

    if eyes_closed:
        blink_state['frames_closed'] += 1
        blink_state['frames_open'] = 0
    else:
        blink_state['frames_open'] += 1
        blink_state['frames_closed'] = 0

    blink_detected = False
    if (not blink_state['eyes_closed_yet']) and (blink_state['frames_closed'] >= CONSEC_FRAMES_CLOSED):
        blink_state['eyes_closed_yet'] = True

    if blink_state['eyes_closed_yet'] and (blink_state['frames_open'] >= CONSEC_FRAMES_OPEN):
        blink_detected = True
        blink_state['eyes_closed_yet'] = False
        blink_state['frames_open'] = 0

    return blink_detected


# -----------------------------------------------------
# TELAS
# -----------------------------------------------------
class HomeScreen(Screen):
    """
    Tela inicial: botões para (Cadastrar, Registrar, Configurações, Logs)
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = BoxLayout(orientation='vertical', spacing=20, padding=20)

        lbl = Label(
            text="IPonto App",
            font_size='26sp',
            size_hint=(1, 0.15)
        )
        layout.add_widget(lbl)

        # Logo
        self.logo = Image(
            # source="C:/Users/itarg/Desktop/face_app/logo_rh247.png", (Imagem Laranja)
            source="C:/Users/itarg/Desktop/face_app/rh247_azul.png", #(Imagem Azul)
            
            size_hint=(1, 0.35),
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(self.logo)

        btn_reg = Button(
            text="Cadastrar Facial",
            font_size="20sp",
            size_hint=(1, 0.12),
            background_color=(0, 0.6, 1, 1),
            color=(1,1,1,1)
        )
        btn_reg.bind(on_press=self.go_register)
        layout.add_widget(btn_reg)

        btn_point = Button(
            text="Registrar Ponto",
            font_size="20sp",
            size_hint=(1, 0.12),
            background_color=(0.2, 0.7, 0.2, 1),
            color=(1,1,1,1)
        )
        btn_point.bind(on_press=self.go_point)
        layout.add_widget(btn_point)

        btn_logs = Button(
            text="Ver Relatórios (Logs)",
            font_size="18sp",
            size_hint=(1, 0.12),
            background_color=(0.5, 0.5, 0.9, 1),
            color=(1,1,1,1)
        )
        btn_logs.bind(on_press=self.go_logs)
        layout.add_widget(btn_logs)

        btn_settings = Button(
            text="Configurações",
            font_size="18sp",
            size_hint=(1, 0.12),
            background_color=(0.4, 0.4, 0.4, 1),
            color=(1,1,1,1)
        )
        btn_settings.bind(on_press=self.go_settings)
        layout.add_widget(btn_settings)

        self.add_widget(layout)

    def go_register(self, instance):
        self.manager.current = 'register_screen'

    def go_point(self, instance):
        self.manager.current = 'point_screen'

    def go_logs(self, instance):
        self.manager.current = 'logs_screen'

    def go_settings(self, instance):
        self.manager.current = 'settings_screen'


class RegisterFaceScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)

        # Layout principal
        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.label_title = Label(text="Cadastrar Facial", font_size='22sp', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.label_title)

        self.camera_widget = Label(text="(camera...)", size_hint=(1, 0.6))
        self.main_layout.add_widget(self.camera_widget)

        self.txt_name = TextInput(
            hint_text="Nome/ID para salvar",
            multiline=False,
            size_hint=(1, 0.1)
        )
        self.main_layout.add_widget(self.txt_name)

        self.btn_capture = Button(
            text="Capturar e Salvar",
            size_hint=(1, 0.1),
            font_size="18sp",
            background_color=(0, 0.6, 1, 1),
            color=(1, 1, 1, 1)
        )
        self.btn_capture.bind(on_press=self.on_capture)
        self.main_layout.add_widget(self.btn_capture)

        btn_back = Button(
            text="Voltar",
            size_hint=(1, 0.1),
            font_size="18sp",
            background_color=(0.5, 0.5, 0.5, 1)
        )
        btn_back.bind(on_press=self.on_back)
        self.main_layout.add_widget(btn_back)

        self.add_widget(self.main_layout)

        self.capture = cv2.VideoCapture(0)
        self.blink_state = {'frames_closed': 0, 'frames_open': 0, 'eyes_closed_yet': False}
        self.blinked = False

        # Corrigindo o erro de método ausente
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        """
        Atualiza a visualização da câmera e processa o feed para detecção de rosto e piscadas.
        """
        if self.manager.current != 'register_screen':
            return

        if not self.capture.isOpened():
            self.camera_widget.text = "(camera indisponível)"
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.camera_widget.text = "(falha na captura)"
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc, loc = get_face_encoding(frame)
        color = (0, 0, 255)

        if enc is not None and loc is not None:
            landmarks = face_landmarks_for_loc(frame_rgb, loc)
            if landmarks is not None:
                blinked_now = detect_blink(landmarks, self.blink_state)
                if blinked_now:
                    self.blinked = True
                color = (0, 255, 0) if self.blinked else (0, 0, 255)

            # Desenha a marcação no rosto
            (top, right, bottom, left) = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Atualiza o botão de captura
        self.btn_capture.disabled = not self.blinked

        # Converter para exibir no Kivy
        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame_disp.tobytes()
        texture = Texture.create(size=(frame_disp.shape[1], frame_disp.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()

        if isinstance(self.camera_widget, Label):
            self.main_layout.remove_widget(self.camera_widget)
            self.camera_widget = Image(size_hint=(1, 0.6))
            self.main_layout.add_widget(self.camera_widget, index=2)

        self.camera_widget.texture = texture




    def on_pre_enter(self, *args):
        if not self.capture.isOpened():
            self.capture.open(0)
        self.reset_state()
        self.label_title.text = "Cadastrar Facial"


    def reset_state(self):
        """
        Reseta o estado da tela de cadastro.
        """
        self.txt_name.text = ""
        self.label_title.text = "Cadastrar Facial"
        self.blink_state = {'frames_closed': 0, 'frames_open': 0, 'eyes_closed_yet': False}
        self.blinked = False

    def on_capture(self, instance):
        """
        Cadastra o rosto somente se não for duplicado.
        """
        name = self.txt_name.text.strip()
        if not name:
            self.label_title.text = "Digite um nome!"
            return

        if not self.blinked:
            self.label_title.text = "Você precisa piscar!"
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.label_title.text = "Falha ao capturar."
            return

        enc, loc = get_face_encoding(frame)
        if enc is None:
            self.label_title.text = "Nenhum rosto detectado!"
            return

        # Tentar salvar o rosto
        try:
            # Salvar o encoding
            save_face_encoding(name, enc)
            
            # Salvar a imagem no diretório
            save_path = f"path_to_face_images/{name}.jpg"
            cv2.imwrite(save_path, frame)

            self.label_title.text = f"Rosto '{name}' cadastrado com sucesso!"
        except ValueError as e:
            self.label_title.text = str(e)


    def on_back(self, instance):
        self.manager.current = 'home_screen'
        self.reset_state()




class PointScreen(Screen):
    """
    Registrar ponto: detecta piscada, nome => salva log
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.label_title = Label(text="Registrar Ponto", font_size='22sp', size_hint=(1,0.1))
        self.main_layout.add_widget(self.label_title)

        self.camera_widget = Label(text="(camera...)", size_hint=(1, 0.6))
        self.main_layout.add_widget(self.camera_widget)

        self.btn_registrar = Button(
            text="Registrar",
            font_size="18sp",
            size_hint=(1,0.1),
            background_color=(0.2,0.7,0.2,1),
            color=(1,1,1,1)
        )
        self.btn_registrar.bind(on_press=self.on_register_point)
        self.main_layout.add_widget(self.btn_registrar)

        btn_back = Button(
            text="Voltar",
            size_hint=(1,0.1),
            font_size="18sp",
            background_color=(0.5,0.5,0.5,1)
        )
        btn_back.bind(on_press=self.on_back)
        self.main_layout.add_widget(btn_back)

        self.add_widget(self.main_layout)

        self.capture = cv2.VideoCapture(0)
        self.blink_state = {'frames_closed':0, 'frames_open':0, 'eyes_closed_yet':False}
        self.blinked = False
        self.recognized_name = None
        self.current_enc = None

        Clock.schedule_interval(self.update, 1.0/30.0)

    def on_pre_enter(self, *args):
        if not self.capture.isOpened():
            self.capture.open(0)
        self.reset_state()

    def reset_state(self):
        self.blink_state = {'frames_closed':0, 'frames_open':0, 'eyes_closed_yet':False}
        self.blinked = False
        self.recognized_name = None
        self.current_enc = None

    def update(self, dt):
        if self.manager.current != 'point_screen':
            return

        if not self.capture.isOpened():
            self.camera_widget.text = "(camera não disponível)"
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.camera_widget.text = "(falha captura)"
            return

        self.current_frame = frame.copy()  # Armazena o quadro atual

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc, loc = get_face_encoding(frame)
        color = (0, 0, 255)

        if enc is not None and loc is not None:
            lands = face_landmarks_for_loc(frame_rgb, loc)
            if lands:
                blinked_now = detect_blink(lands, self.blink_state)
                if blinked_now:
                    self.blinked = True
                    who = compare_with_db(enc, self.current_frame)
                    if who is None:
                        self.label_title.text = "Nenhum registro encontrado."
                    else:
                        self.recognized_name = who
                        self.label_title.text = f"Bem-vindo(a), {who}!"
                    self.current_enc = enc

                if self.blinked and self.recognized_name:
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 0) if (self.blinked and self.recognized_name) else (0, 0, 255)

            (top, right, bottom, left) = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            if self.blinked and self.recognized_name:
                cv2.putText(frame, self.recognized_name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        self.btn_registrar.disabled = not (self.blinked and self.recognized_name)

        # Atualiza a câmera
        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame_disp.tobytes()
        texture = Texture.create(
            size=(frame_disp.shape[1], frame_disp.shape[0]),
            colorfmt='rgb'
        )
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()

        if isinstance(self.camera_widget, Label):
            self.main_layout.remove_widget(self.camera_widget)
            self.camera_widget = Image(size_hint=(1, 0.6))
            self.main_layout.add_widget(self.camera_widget, index=2)

        self.camera_widget.texture = texture



    def on_register_point(self, instance):
        if self.current_enc is not None:
            recognized_name = compare_with_db(self.current_enc, self.current_frame)
            if recognized_name:
                save_log_punch(recognized_name, self.current_enc)
                self.label_title.text = f"Ponto Registrado, {recognized_name}!"
            else:
                self.label_title.text = "Nenhum registro encontrado."
        else:
            self.label_title.text = "Nenhum rosto detectado."



    def on_back(self, instance):
        self.manager.current = 'home_screen'


class LogsScreen(Screen):
    """
    Tela para listar logs (face_logs), exportar CSV, sincronizar com servidor.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        lbl = Label(text="Relatórios (Logs)", font_size='22sp', size_hint=(1,0.1))
        self.main_layout.add_widget(lbl)

        # ScrollView para exibir logs
        self.scroll = ScrollView(size_hint=(1,0.7))
        self.logs_container = BoxLayout(orientation='vertical', size_hint_y=None)
        self.logs_container.bind(minimum_height=self.logs_container.setter('height'))
        self.scroll.add_widget(self.logs_container)
        self.main_layout.add_widget(self.scroll)

        # Botões de ação (exportar, sync)
        btns_layout = BoxLayout(orientation='horizontal', size_hint=(1,0.1))

        self.btn_export = Button(text="Export CSV", size_hint=(0.5,1))
        self.btn_export.bind(on_press=self.on_export_csv)
        btns_layout.add_widget(self.btn_export)

        self.btn_sync = Button(text="Sincronizar", size_hint=(0.5,1))
        self.btn_sync.bind(on_press=self.on_sync)
        btns_layout.add_widget(self.btn_sync)

        self.main_layout.add_widget(btns_layout)

        # Botão voltar
        btn_back = Button(text="Voltar", size_hint=(1,0.1), background_color=(0.5,0.5,0.5,1))
        btn_back.bind(on_press=self.on_back)
        self.main_layout.add_widget(btn_back)

        self.add_widget(self.main_layout)

    def on_pre_enter(self, *args):
        self.load_logs()

    def load_logs(self):
        """
        Carrega a lista de logs e exibe na tela como uma tabela estilizada.
        """
        self.logs_container.clear_widgets()

        # Criar layout da tabela
        table = GridLayout(cols=4, size_hint_y=None, padding=[10, 10, 10, 10], spacing=5)
        table.bind(minimum_height=table.setter('height'))

        # Adicionar cabeçalho da tabela
        headers = ["ID", "Nome", "Data/Hora", "Tipo"]
        for header in headers:
            header_label = Label(
                text=header,
                color=(1, 1, 1, 1),
                bold=True,
                size_hint_y=None,
                height=40,
                halign='center',
                valign='middle'
            )
            header_label.bind(size=header_label.setter('text_size'))
            table.add_widget(header_label)

        # Carregar os logs do banco de dados
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, name, date_time, log_type FROM face_logs ORDER BY id DESC')
        logs = c.fetchall()
        conn.close()

        for log_id, name, dt, log_type in logs:
            # Garantir que os valores não sejam None
            log_id = log_id if log_id is not None else "N/A"
            name = name if name is not None else "N/A"
            dt = dt if dt is not None else "N/A"
            log_type = log_type if log_type is not None else "N/A"

            # Define a cor para o tipo (Entrada ou Saída)
            bg_color = (0.2, 0.8, 0.2, 1) if log_type == "Entrada" else (0.8, 0.2, 0.2, 1)

            # Adiciona os dados na tabela
            for value in [str(log_id), name, dt, log_type]:
                cell_label = Label(
                    text=value,
                    color=(1, 1, 1, 1),
                    size_hint_y=None,
                    height=40,
                    halign='center',
                    valign='middle'
                )
                cell_label.bind(size=cell_label.setter('text_size'))
                table.add_widget(cell_label)

        # Adiciona a tabela ao container
        self.logs_container.add_widget(table)




    def on_export_csv(self, instance):
        """
        Exportar os logs para CSV local.
        Evitar travar a UI -> usar thread
        """
        def export_csv_thread():
            rows = load_all_logs()
            csv_path = "face_logs_export.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("id,name,date_time,encoding\n")
                for (log_id, name, dt, enc) in rows:
                    f.write(f"{log_id},{name},{dt},{enc}\n")
            self.update_label_main_thread("Export concluído")

        th = threading.Thread(target=export_csv_thread)
        th.start()

    @mainthread
    def update_label_main_thread(self, msg):
        self.btn_export.text = msg

    def on_sync(self, instance):
        """
        Exemplo: Sincronizar com servidor via HTTP POST.
        """
        def sync_thread():
            url = "https://api.meuservidor.com/pontos"
            logs = load_all_logs()
            payload = []
            for (log_id, name, dt, enc) in logs:
                payload.append({
                    "id": log_id,
                    "name": name,
                    "date_time": dt,
                    "encoding": enc
                })

            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    self.update_label_main_thread("Sincronização OK!")
                else:
                    self.update_label_main_thread(f"Erro sync: {resp.status_code}")
            except Exception as e:
                self.update_label_main_thread(f"Falha sync: {e}")

        th = threading.Thread(target=sync_thread)
        th.start()

    def on_back(self, instance):
        self.manager.current = 'home_screen'


class SettingsScreen(Screen):
    """
    Tela de configurações: tolerância face, EAR_THRESHOLD, etc.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        lbl = Label(text="Configurações", font_size='22sp', size_hint=(1,0.1))
        layout.add_widget(lbl)

        # Tolerância
        self.tolerance_input = TextInput(
            text=str(FACE_RECOG_TOLERANCE),
            hint_text="Face Tolerance (0.6 default)",
            multiline=False,
            size_hint=(1,0.1),
            input_filter="float"
        )
        layout.add_widget(self.tolerance_input)

        # EAR
        self.ear_input = TextInput(
            text=str(EAR_THRESHOLD),
            hint_text="EAR threshold (0.22 default)",
            multiline=False,
            size_hint=(1,0.1),
            input_filter="float"
        )
        layout.add_widget(self.ear_input)

        # Botão Salvar
        btn_save = Button(text="Salvar", size_hint=(1,0.1), font_size="18sp")
        btn_save.bind(on_press=self.on_save)
        layout.add_widget(btn_save)

        # Botão voltar
        btn_back = Button(text="Voltar", size_hint=(1,0.1), background_color=(0.5,0.5,0.5,1))
        btn_back.bind(on_press=self.on_back)
        layout.add_widget(btn_back)

        self.add_widget(layout)

    def on_pre_enter(self, *args):
        global EAR_THRESHOLD, FACE_RECOG_TOLERANCE
        # Carrega valores atuais
        self.tolerance_input.text = str(FACE_RECOG_TOLERANCE)
        self.ear_input.text = str(EAR_THRESHOLD)

    def on_save(self, instance):
        global FACE_RECOG_TOLERANCE, EAR_THRESHOLD
        try:
            FACE_RECOG_TOLERANCE = float(self.tolerance_input.text)
            EAR_THRESHOLD = float(self.ear_input.text)
        except ValueError:
            pass
        self.manager.current = 'home_screen'

    def on_back(self, instance):
        self.manager.current = 'home_screen'


# -----------------------------------------------------
# APP PRINCIPAL
# -----------------------------------------------------
class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(HomeScreen(name='home_screen'))
        self.add_widget(RegisterFaceScreen(name='register_screen'))
        self.add_widget(PointScreen(name='point_screen'))
        self.add_widget(LogsScreen(name='logs_screen'))
        self.add_widget(SettingsScreen(name='settings_screen'))


class FaceApp(App):
    def build(self):
        self.title = "IPontoApp"
        create_db_if_not_exists()           # Cria as tabelas, se necessário
        ensure_encoding_hash_column()       # Garante que a coluna exista
        update_existing_encodings_with_hash()  # Atualiza registros antigos
        update_db_schema()                  # Outros ajustes no banco
        return MyScreenManager()




if __name__ == "__main__":
    FaceApp().run()
