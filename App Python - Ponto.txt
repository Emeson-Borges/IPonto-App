"""
Exemplo de App Kivy APRIMORADO com:
- Cadastro e Registro de Ponto (piscada)
- Tabela de logs (face_logs)
- Tela de Relatórios (LogsScreen) para listar e exportar
- Tela de Configurações (SettingsScreen) para EAR, Tolerância etc.
- Exemplo de "Sincronizar" logs via POST a um servidor (simulado)
- Uso de threading para evitar travar a UI

Corrigido para não dar erro ValueError ao trocar Label -> Image 
nas telas de cadastro e registro de ponto.
"""

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

import cv2
import face_recognition
import numpy as np
import sqlite3
import json
import os
import math

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
FACE_RECOG_TOLERANCE = 0.6  # Ajustável

# -----------------------------------------------------
# BANCO DE DADOS
# -----------------------------------------------------
def create_db_if_not_exists():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS face_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            date_time TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def user_already_exists(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM face_encodings WHERE name=?', (name,))
    count = c.fetchone()[0]
    conn.close()
    return (count > 0)

def save_face_encoding(name, encoding):
    encoding_json = json.dumps(encoding.tolist())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO face_encodings (name, encoding) VALUES (?, ?)', (name, encoding_json))
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
    enc_json = json.dumps(encoding.tolist())
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO face_logs (name, encoding, date_time) VALUES (?, ?, ?)',
              (name, enc_json, now_str))
    conn.commit()
    conn.close()

def load_all_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, date_time, encoding FROM face_logs ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return rows

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
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    loc = detect_main_face(frame_rgb)
    if loc is None:
        return None, None
    encs = face_recognition.face_encodings(frame_rgb, known_face_locations=[loc], num_jitters=1)
    if len(encs) == 0:
        return None, None
    return encs[0], loc

def compare_with_db(unknown_enc):
    known_faces = load_face_encodings()
    if not known_faces:
        return None

    best_name = None
    best_dist = 999
    for name, arr_list in known_faces.items():
        for arr in arr_list:
            dist = face_recognition.face_distance([arr], unknown_enc)[0]
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist < FACE_RECOG_TOLERANCE:
        return best_name
    return None

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
    left_eye = landmarks.get('left_eye')
    right_eye = landmarks.get('right_eye')
    if not left_eye or not right_eye:
        return False

    left_ear = calc_ear(left_eye)
    right_ear = calc_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    eyes_closed = (avg_ear < EAR_THRESHOLD)

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
            source="C:/Users/itarg/Desktop/face_app/logo_rh247.png",
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
    """
    Tela de cadastro (exige piscada).
    """
    def __init__(self, **kw):
        super().__init__(**kw)

        # Salvar em self.main_layout para podermos trocar o Label pela Image
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
            color=(1,1,1,1)
        )
        self.btn_capture.bind(on_press=self.on_capture)
        self.main_layout.add_widget(self.btn_capture)

        btn_back = Button(
            text="Voltar",
            size_hint=(1, 0.1),
            font_size="18sp",
            background_color=(0.5,0.5,0.5,1)
        )
        btn_back.bind(on_press=self.on_back)
        self.main_layout.add_widget(btn_back)

        self.add_widget(self.main_layout)

        self.capture = cv2.VideoCapture(0)
        self.blink_state = {'frames_closed':0, 'frames_open':0, 'eyes_closed_yet':False}
        self.blinked = False

        Clock.schedule_interval(self.update, 1.0/30.0)

    def on_pre_enter(self, *args):
        if not self.capture.isOpened():
            self.capture.open(0)
        self.reset_state()

    def reset_state(self):
        self.txt_name.text = ""
        self.blink_state = {'frames_closed':0, 'frames_open':0, 'eyes_closed_yet':False}
        self.blinked = False

    def update(self, dt):
        if self.manager.current != 'register_screen':
            return

        if not self.capture.isOpened():
            self.camera_widget.text = "(camera indisponível)"
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.camera_widget.text = "(falha na captura)"
            return

        # Processar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc, loc = get_face_encoding(frame)
        color = (0,0,255)

        if enc is not None and loc is not None:
            lands = face_landmarks_for_loc(frame_rgb, loc)
            if lands is not None:
                blinked_now = detect_blink(lands, self.blink_state)
                if blinked_now:
                    self.blinked = True
                color = (0,255,0) if self.blinked else (0,0,255)

            (top, right, bottom, left) = loc
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        self.btn_capture.disabled = not self.blinked

        # Converter p/ exibir
        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame_disp.tobytes()
        texture = Texture.create(size=(frame_disp.shape[1], frame_disp.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()

        # Se ainda for Label, trocamos no self.main_layout
        if isinstance(self.camera_widget, Label):
            try:
                parent_idx = self.main_layout.children.index(self.camera_widget)
            except ValueError:
                parent_idx = 0

            self.main_layout.remove_widget(self.camera_widget)
            self.camera_widget = Image(size_hint=(1,0.6))
            # Reinsere no mesmo "index" invertido
            self.main_layout.add_widget(
                self.camera_widget, 
                index=len(self.main_layout.children) - parent_idx
            )

        self.camera_widget.texture = texture

    def on_capture(self, instance):
        """
        Cadastra se piscou e não existe no DB.
        """
        name = self.txt_name.text.strip()
        if not name:
            self.label_title.text = "Digite um nome!"
            return
        if user_already_exists(name):
            self.label_title.text = f"'{name}' já está cadastrado!"
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

        save_face_encoding(name, enc)
        self.label_title.text = f"Rosto '{name}' cadastrado!"

    def on_back(self, instance):
        self.manager.current = 'home_screen'


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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc, loc = get_face_encoding(frame)
        color = (0,0,255)

        if enc is not None and loc is not None:
            lands = face_landmarks_for_loc(frame_rgb, loc)
            if lands:
                blinked_now = detect_blink(lands, self.blink_state)
                if blinked_now:
                    self.blinked = True
                    who = compare_with_db(enc)
                    self.recognized_name = who
                    self.current_enc = enc

                if self.blinked and self.recognized_name:
                    color = (0,255,0)
                else:
                    color = (0,255,0) if (self.blinked and self.recognized_name) else (0,0,255)

            (top,right,bottom,left) = loc
            cv2.rectangle(frame, (left,top),(right,bottom), color,2)
            if self.blinked and self.recognized_name:
                cv2.putText(frame, self.recognized_name, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)

        self.btn_registrar.disabled = not (self.blinked and self.recognized_name)

        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame_disp.tobytes()
        texture = Texture.create(
            size=(frame_disp.shape[1], frame_disp.shape[0]),
            colorfmt='rgb'
        )
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()

        # Se camera_widget ainda for Label, trocamos no self.main_layout
        if isinstance(self.camera_widget, Label):
            try:
                parent_idx = self.main_layout.children.index(self.camera_widget)
            except ValueError:
                parent_idx = 0

            self.main_layout.remove_widget(self.camera_widget)
            self.camera_widget = Image(size_hint=(1,0.6))
            self.main_layout.add_widget(
                self.camera_widget, 
                index=len(self.main_layout.children) - parent_idx
            )

        self.camera_widget.texture = texture

    def on_register_point(self, instance):
        if self.recognized_name and self.current_enc is not None:
            save_log_punch(self.recognized_name, self.current_enc)
            self.label_title.text = f"Bem-vindo(a), {self.recognized_name}!"
        else:
            self.label_title.text = "Rosto não reconhecido."

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
        Carrega a lista de logs e exibe na tela.
        """
        self.logs_container.clear_widgets()
        rows = load_all_logs()
        for (log_id, name, dt, enc) in rows:
            label_txt = f"ID:{log_id} | {name} | {dt}"
            lb = Label(text=label_txt, size_hint_y=None, height=40)
            self.logs_container.add_widget(lb)

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
        create_db_if_not_exists()
        return MyScreenManager()


if __name__ == "__main__":
    FaceApp().run()
