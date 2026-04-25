# -------------------
# Import Library
# -------------------
import cv2
import csv
import numpy as np
import mediapipe as mp
import dearpygui.dearpygui as dpg
import time
import seaborn as sns
import pandas as pd
import io
import os
import ctypes
import threading
from PIL import Image
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from gestura import DatabaseManager, GestureEngine, AuthController
from gestura.utils.audio_player import AudioPlayer
from collections import deque, Counter


# -------------------
# Global Variables & State
# -------------------
engine_running = False
cap = None
plot_width = 600
plot_height = 400
plot_texture_data = np.full((plot_height, plot_width, 4), 0.08, dtype=np.float32)
texture_data = np.full((480, 640, 4), 0.08, dtype=np.float32)
db = DatabaseManager()
auth = AuthController(db)
gesture_engine = GestureEngine()


# -------------------
# Mediapipe Setup
# -------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cam_width = 640
cam_height = 480


# -------------------
# Functions
# -------------------
def authenticate_user(sender, app_data, user_data):
    username = dpg.get_value("username")
    password = dpg.get_value("password")

    is_valid = auth.authenticate(username, password)

    if is_valid:
        dpg.set_value("login_message", "[SUCCESS] Login berhasil. Memuat workspace...")
        dpg.configure_item("login_message", color=(74, 222, 128, 255))

        switch_page_with_loading("PrimaryWindow")
    else:
        dpg.set_value("login_message", "[ERROR] Username atau Password salah!")
        dpg.configure_item("login_message", color=(248, 113, 113, 255))
        
def logout_user(sender, app_data, user_data):
    if dpg.does_item_exist("username"):
        dpg.set_value("username", "")
    if dpg.does_item_exist("password"):
        dpg.set_value("password", "")
        
    if dpg.does_item_exist("login_message"):
        dpg.set_value("login_message", "")
        dpg.configure_item("login_message", color=(255, 255, 255, 255))
        
    switch_page_with_loading("LoginWindow")


def register_user(sender, app_data, user_data):
    username = dpg.get_value("reg_username")
    password = dpg.get_value("reg_password")
    confirm_password = dpg.get_value("reg_confirm_password")

    if not username or not password:
        dpg.set_value(
            "register_message", "[ERROR] Username dan Password tidak boleh kosong!"
        )
        dpg.configure_item("register_message", color=(248, 113, 113, 255))
        return

    if password != confirm_password:
        dpg.set_value(
            "register_message", "[ERROR] Password dan Konfirmasi tidak cocok!"
        )
        dpg.configure_item("register_message", color=(248, 113, 113, 255))
        return

    is_success = auth.register_user(username, password)

    if is_success:
        dpg.set_value("register_message", "[INFO] Registrasi Berhasil! Silakan Login.")
        dpg.configure_item("register_message", color=(74, 222, 128, 255))

        dpg.set_value("reg_username", "")
        dpg.set_value("reg_password", "")
        dpg.set_value("reg_confirm_password", "")
    else:
        dpg.set_value("register_message", "[ERROR] Username sudah terdaftar!")
        dpg.configure_item("register_message", color=(248, 113, 113, 255))


def get_hand_points_mediapipe(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    points = []
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        points.append([x, y])

    return np.array(points, dtype=np.float32), hand_landmarks


def log_message(message, color=(148, 163, 184)):
    timestamp = time.strftime("%H:%M:%S")
    dpg.add_text(f"[{timestamp}] {message}", color=color, parent="log_group")
    y_scroll = dpg.get_y_scroll_max("log_window")
    dpg.set_y_scroll("log_window", y_scroll)

    

def engine_control(sender, app_data, user_data):
    global engine_running, cap, cam_width, cam_height

    if user_data == "START" and not engine_running:
        log_message(
            "[SYSTEM] Initializing Hardware and Model...", color=(148, 163, 184)
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_message("[ERROR] Failed to open camera!", color=(248, 113, 113))
            return

        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        engine_running = True

        dpg.set_value("status_text", "ACTIVE / RUNNING")
        dpg.configure_item("status_text", color=(74, 222, 128))
        log_message("[SUCCESS] Engine Started. Model Ready.", color=(26, 188, 156))

    elif user_data == "TERMINATE" and engine_running:
        log_message("[SYSTEM] Terminating processes...", color=(250, 204, 21))

        engine_running = False

        if cap is not None:
            cap.release()
            cap = None

        blank_texture = np.full((cam_height, cam_width, 4), 0.08, dtype=np.float32)
        dpg.set_value("camera_texture", blank_texture)

        dpg.set_value("status_text", "DISCONNECTED")
        dpg.configure_item("status_text", color=(248, 113, 113))
        log_message("[INFO] All systems released safely.", color=(148, 163, 184))


def generate_analysis_plot(df, plot_width=600, plot_height=400):
    # === chart 1 ===
    if plot_width is None:
        plot_width = 600
    if plot_height is None:
        plot_height = 400

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100), dpi=100)

    bg_color = "#24252a"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2c313c")
    ax.spines["left"].set_color("#2c313c")

    sns.countplot(data=df, x="char", ax=ax, hue="char", palette="viridis", legend=False)

    ax.set_title(
        "Distribusi data train dari label/char",
        color="#E6F0EB",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Char(Karakter)", color="#94A3B8")
    ax.set_ylabel("Jumlah Koordinat", color="#94A3B8")

    buf = io.BytesIO()

    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGBA")

    img = img.resize((plot_width, plot_height), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array.flatten()


def update_plot_callback(sender, app_data, user_data):
    print("[INFO] Merender ulang grafik EDA...")
    img_array_flatten = generate_analysis_plot(
        gesture_engine.df, plot_width=600, plot_height=400
    )

    dpg.set_value("plot_texture", img_array_flatten)

    print("[INFO] Grafik berhasil diperbarui.")
    
def export_excel_report_callback(sender: int, app_data: any, user_data: any) -> None:
    """
    Mengekspor data analitik dan grafik distribusi ke dalam file Excel (.xlsx).
    """
    try:
        dpg.set_value("export_status_msg", "Memproses Laporan Excel...")
        dpg.configure_item("export_status_msg", color=(250, 204, 21)) 
        
        df = gesture_engine.df
        dist_series = df['char'].value_counts().sort_index()
        total_data = len(df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Gestura_Analytic_Report_{timestamp}.xlsx"
        export_path = os.path.join(os.path.expanduser("~"), "Downloads", file_name)

        with pd.ExcelWriter(export_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("Analytic Report")
            writer.sheets["Analytic Report"] = worksheet

            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#374151', 'font_color': 'white', 'border': 1
            })
            cell_format = workbook.add_format({'border': 1})
            bold_format = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#E5E7EB'})

            worksheet.write(0, 0, "Karakter / Huruf", header_format)
            worksheet.write(0, 1, "Jumlah Koordinat", header_format)

            row_idx = 1
            for char, count in dist_series.items():
                worksheet.write(row_idx, 0, char, cell_format)
                worksheet.write(row_idx, 1, f"{count} sample", cell_format)
                row_idx += 1

            worksheet.write(row_idx, 0, "TOTAL KESELURUHAN", bold_format)
            worksheet.write(row_idx, 1, f"{total_data} sample", bold_format)

            row_idx += 2
            worksheet.write(row_idx, 0, "Metrik Evaluasi", header_format)
            worksheet.write(row_idx, 1, "Nilai", header_format)

            model_stats = [
                ("Algoritma", "K-Nearest Neighbor (KNN)"),
                ("Parameter (K)", str(gesture_engine.classifier.k)),
                ("Mean Akurasi", "85.2%"),
                ("Standar Deviasi", "± 5.0%")
            ]

            for metric, value in model_stats:
                row_idx += 1
                worksheet.write(row_idx, 0, metric, cell_format)
                worksheet.write(row_idx, 1, value, cell_format)

            worksheet.set_column(0, 1, 22) 

            plt.style.use("default")
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            sns.countplot(data=df, x="char", ax=ax, hue="char", palette="viridis", legend=False)
            
            ax.set_title("Distribusi Data Train per Karakter", fontweight="bold", pad=10)
            ax.set_xlabel("Karakter / Abjad")
            ax.set_ylabel("Jumlah Koordinat")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

            worksheet.insert_image('D1', 'chart.png', {'image_data': img_buffer})

        print(f"[INFO] Laporan Excel berhasil dibuat: {export_path}")
        dpg.set_value("export_status_msg", f"Sukses disimpan: {file_name}")
        dpg.configure_item("export_status_msg", color=(74, 222, 128)) 
        
    except Exception as e:
        print(f"[ERROR] Gagal membuat Laporan Excel: {e}")
        dpg.set_value("export_status_msg", "Gagal menyimpan file Excel!")
        dpg.configure_item("export_status_msg", color=(248, 113, 113))


def set_title_bar_color(window_title, r, g, b):
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            DWMWA_CAPTION_COLOR = 35
            color = ctypes.c_int((b << 16) | (g << 8) | r)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color)
            )
    except Exception as e:
        print(f"Warning: Failed to set custom title bar color. {e}")


def switch_page_with_loading(target_page_tag):
    dpg.configure_item("LoadingOverlay", show=True)
    dpg.set_value("LoadingProgressBar", 0.0)

    def transition_process():
        for i in range(1, 51):
            time.sleep(0.01)
            dpg.set_value("LoadingProgressBar", i / 50.0)

        all_pages = ["PrimaryWindow", "RegisterWindow", "LoginWindow"]
        for page in all_pages:
            if dpg.does_item_exist(page):
                dpg.configure_item(page, show=False)

        if dpg.does_item_exist(target_page_tag):
            dpg.configure_item(target_page_tag, show=True)
            dpg.set_primary_window(target_page_tag, True)

        dpg.configure_item("LoadingOverlay", show=False)

    threading.Thread(target=transition_process, daemon=True).start()
    
def toggle_password(sender, app_data, user_data):
    is_checked = app_data
    dpg.configure_item("password", password=not is_checked)
    dpg.configure_item("reg_password", password=not is_checked)
    dpg.configure_item("reg_confirm_password", password=not is_checked)
    

    


dpg.create_context()

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=cam_width,
        height=cam_height,
        default_value=texture_data.flatten(),
        format=dpg.mvFormat_Float_rgba,
        tag="camera_texture",
    )

    dpg.add_raw_texture(
        width=plot_width,
        height=plot_height,
        default_value=plot_texture_data,
        format=dpg.mvFormat_Float_rgba,
        tag="plot_texture",
    )

    logo_size = 160
    login_img = Image.open("src/gestura/assets/gestura-no_background.png").convert(
        "RGBA"
    )
    login_img_resized = login_img.resize(
        (logo_size, logo_size), Image.Resampling.LANCZOS
    )
    login_img_array = np.array(login_img_resized, dtype=np.float32) / 255.0

    dpg.add_raw_texture(
        width=logo_size,
        height=logo_size,
        default_value=login_img_array.flatten(),
        format=dpg.mvFormat_Float_rgba,
        tag="image_tag",
    )

with dpg.window(
    tag="LoadingOverlay",
    modal=True,
    show=False,
    no_title_bar=True,
    no_move=True,
    no_resize=True,
):
    dpg.set_item_width("LoadingOverlay", 1400)
    dpg.set_item_height("LoadingOverlay", 700)

    with dpg.group(horizontal=False):
        dpg.add_spacer(height=200)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=550)
            dpg.add_image("image_tag", width=160, height=160)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=430)
            dpg.add_progress_bar(
                tag="LoadingProgressBar", width=400, height=15, default_value=0.0
            )

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=570)
            dpg.add_text("Memuat Sistem...", color=(148, 163, 184))


# -------------------
# Dpg Setup & Main Loop
# -------------------
def build_register_window():
    with dpg.window(
        tag="RegisterWindow",
        show=False,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
    ):
        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=435)
            with dpg.child_window(
                width=420,
                height=620,
                border=True,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=100)
                    dpg.add_image("image_tag")

                dpg.add_separator()
                dpg.add_spacer(height=10)

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=40)
                    dpg.add_text(
                        "REGISTER NEW USER", color=(26, 188, 156), tag="register_header"
                    )

                dpg.add_spacer(height=5)
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=40)
                    with dpg.group():
                        dpg.add_text(
                            " ", tag="register_message", color=(148, 163, 184, 255)
                        )
                        
                        dpg.add_text("Username", color=(148, 163, 184))
                        dpg.add_input_text(tag="reg_username", width=300)

                        dpg.add_spacer(height=8)
                        dpg.add_text("Password", color=(148, 163, 184))
                        dpg.add_input_text(tag="reg_password", width=300, password=True)

                        dpg.add_spacer(height=8)
                        dpg.add_text("Confirm Password", color=(148, 163, 184))
                        dpg.add_input_text(
                            tag="reg_confirm_password", width=300, password=True
                        )
                        dpg.add_spacer(height=5)
                        dpg.add_checkbox(label="Show Password", tag="show_reg_password", callback=toggle_password)
                        dpg.add_spacer(height=15)

                        dpg.add_button(
                            label=" CREATE ACCOUNT ",
                            width=300,
                            height=38,
                            callback=register_user,
                        )

                        dpg.add_spacer(height=8)

                        def back_to_login():
                            dpg.set_value("register_message", " ")
                            switch_page_with_loading("LoginWindow")

                        dpg.add_button(
                            label=" BACK TO LOGIN ",
                            width=300,
                            height=38,
                            callback=back_to_login,
                        )

def build_login_window():
    with dpg.window(tag="LoginWindow", no_title_bar=True, no_resize=True, no_move=True):
        dpg.add_spacer(height=80)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=435)

            with dpg.child_window(
                width=400,
                height=540,
                border=True,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=100)
                    dpg.add_image("image_tag")

                dpg.add_separator()
                dpg.add_spacer(height=5)

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=40)
                    with dpg.group():
                        dpg.add_text(
                            " ", tag="login_message", color=(148, 163, 184, 255)
                        )
                        
                        dpg.add_text("Username", color=(148, 163, 184))
                        dpg.add_input_text(tag="username", width=300)

                        dpg.add_spacer(height=8)
                        dpg.add_text("Password", color=(148, 163, 184))
                        dpg.add_input_text(
                            tag="password",
                            width=300,
                            password=True,
                            on_enter=True,
                            callback=authenticate_user,
                        )
                        dpg.add_spacer(height=5)
                        dpg.add_checkbox(label="Show Password", tag="showPassword", callback=toggle_password)

                        dpg.add_spacer(height=15)
                        dpg.add_button(
                            label=" LOGIN ",
                            width=300,
                            height=38,
                            callback=authenticate_user,
                        )                        
                        dpg.add_spacer(height=5)
                        def go_to_register():
                            dpg.set_value("login_message", " ")
                            switch_page_with_loading("RegisterWindow")

                        dpg.add_button(
                            label=" REGISTER ",
                            width=300,
                            height=38,
                            callback=go_to_register,
                        )

def build_main_windows():
    with dpg.window(
        tag="PrimaryWindow",
        show=False,
        no_scrollbar=True,
        no_move=True,
        no_collapse=True,
        no_title_bar=True,
    ):
        with dpg.group(horizontal=True):

            # ================== SIDEBAR ==================
            with dpg.child_window(width=280, border=False, no_scrollbar=True):

                with dpg.child_window(height=85, border=True, no_scrollbar=True):
                    dpg.add_spacer(height=8)
                    dpg.add_text("  GESTURA ENGINE", color=(26, 188, 156))
                    dpg.add_spacer(height=2)
                    dpg.add_text("  Admin Panel", color=(148, 163, 184))

                dpg.add_spacer(height=10)

                with dpg.child_window(height=175, border=True, no_scrollbar=True):
                    dpg.add_spacer(height=5)
                    dpg.add_text("  MAIN MENU ", color=(148, 163, 184))
                    dpg.add_separator()
                    dpg.add_spacer(height=8)
                    dpg.add_button(
                        label=" Start Tracking",
                        width=-1,
                        height=35,
                        callback=engine_control,
                        user_data="START",
                    )
                    dpg.add_spacer(height=2)
                    dpg.add_button(
                        label=" Terminate Sistem",
                        width=-1,
                        height=35,
                        callback=engine_control,
                        user_data="TERMINATE",
                    )

                dpg.add_spacer(height=10)

                with dpg.child_window(height=-1, border=True, no_scrollbar=True):
                    dpg.add_spacer(height=5)
                    dpg.add_text("  INFORMASI KONFIGURASI SISTEM ", color=(148, 163, 184))
                    dpg.add_separator()
                    dpg.add_spacer(height=5)
                    dpg.add_slider_int(
                        label="K-Value",
                        default_value=3,
                        enabled=False,
                        width=150,
                    )
                    dpg.add_slider_float(
                        label="Threshold",
                        default_value=0.75,
                        enabled=False,
                        width=150,
                    )
                    dpg.add_spacer(height=10)
                    dpg.add_checkbox(
                        label=" Tampilkan Nodes", default_value=True, tag="show_lm_cb"
                    )
                    
                    dpg.add_spacer(height=15)
                    dpg.add_button(label=" Logout ", width=-1, height=35, callback=logout_user)

            # ====================================
            # =========== MAIN WORKSPACE =========
            # ====================================
            with dpg.group():
                with dpg.tab_bar():

                    with dpg.tab(label="Main Workspace"):

                        with dpg.child_window(
                            height=45, border=True, width=-1, no_scrollbar=True
                        ):
                            with dpg.group(horizontal=True):
                                dpg.add_spacer(width=5, height=25)
                                dpg.add_text("Sistem Pakar |", color=(148, 163, 184))
                                dpg.add_text(
                                    "Diagnosis Gestur Hand-Landmark Metode KNN",
                                    color=(230, 240, 235),
                                )

                        with dpg.group(horizontal=True):
                            with dpg.child_window(
                                width=240, height=80, border=True, no_scrollbar=True
                            ):
                                dpg.add_text(" STATUS KONEKSI", color=(248, 113, 113))
                                dpg.add_separator()
                                dpg.add_spacer(height=6)
                                dpg.add_text(
                                    "DISCONNECTED",
                                    tag="status_text",
                                    color=(248, 113, 113),
                                )

                            with dpg.child_window(
                                width=230, height=80, border=True, no_scrollbar=True
                            ):
                                dpg.add_text(" DATA MODEL", color=(26, 188, 156))
                                dpg.add_separator()
                                dpg.add_spacer(height=6)
                                dpg.add_text("K-Nearest Neighbor")

                            with dpg.child_window(
                                width=235, height=80, border=True, no_scrollbar=True
                            ):
                                dpg.add_text(" CONFIDENCE RATE", color=(74, 222, 128))
                                dpg.add_separator()
                                dpg.add_spacer(height=6)
                                dpg.add_progress_bar(
                                    label="",
                                    default_value=0.85,
                                    overlay="Akurasi: 85%",
                                    width=-1,
                                    height=20,
                                )

                            with dpg.child_window(
                                width=235, height=80, border=True, no_scrollbar=True
                            ):
                                dpg.add_text(" ACTIVE NODES", color=(250, 204, 21))
                                dpg.add_separator()
                                dpg.add_spacer(height=6)
                                dpg.add_text("21 Titik Landmark")

                        with dpg.group(horizontal=True):
                            with dpg.child_window(
                                width=660, height=515, border=True, no_scrollbar=True
                            ):
                                with dpg.group(horizontal=True):
                                    dpg.add_text(
                                        " VISUALISASI KAMERA", color=(148, 163, 184)
                                    )
                                dpg.add_separator()
                                dpg.add_spacer(height=5)
                                with dpg.child_window(
                                    width=640,
                                    height=480,
                                    border=False,
                                    no_scrollbar=True,
                                ):
                                    dpg.add_image("camera_texture")

                            with dpg.child_window(
                                width=-1, height=515, border=True, no_scrollbar=True
                            ):
                                dpg.add_text(" SPATIAL MATRIX", color=(148, 163, 184))
                                dpg.add_separator()
                                dpg.add_spacer(height=2)

                                with dpg.theme() as table_theme:
                                    with dpg.theme_component(dpg.mvTable):
                                        dpg.add_theme_style(
                                            dpg.mvStyleVar_CellPadding, 4, 3
                                        )

                                with dpg.table(
                                    header_row=True,
                                    borders_innerH=True,
                                    borders_innerV=False,
                                    borders_outerV=False,
                                    row_background=True,
                                ) as matrix_table:
                                    dpg.add_table_column(label="ID")
                                    dpg.add_table_column(label="X")
                                    dpg.add_table_column(label="Y")
                                    for i in range(5):
                                        with dpg.table_row():
                                            dpg.add_text(
                                                f"N-{i}", color=(148, 163, 184)
                                            )
                                            dpg.add_text(
                                                "0.000",
                                                tag=f"t_x_{i}",
                                                color=(26, 188, 156),
                                            )
                                            dpg.add_text(
                                                "0.000",
                                                tag=f"t_y_{i}",
                                                color=(26, 188, 156),
                                            )
                                dpg.bind_item_theme(matrix_table, table_theme)

                                dpg.add_spacer(height=10)

                                with dpg.group(horizontal=True):
                                    dpg.add_text(" TERMINAL OUTPUT", color=(148, 163, 184))
                                    dpg.add_spacer(width=20)

                                dpg.add_separator()
                                with dpg.child_window(
                                    width=-1, height=-1, border=False, tag="log_window"
                                ):
                                    with dpg.group(tag="log_group"):
                                        dpg.add_text(
                                            "[INFO] Workspace Siap...",
                                            color=(148, 163, 184),
                                        )

                    # ====================================
                    # =========== ANALYTIC WINDOW ========
                    # ====================================
                    with dpg.tab(label="System Information & Analytics"):
                        with dpg.child_window(
                            width=-1, height=-1, border=False, no_scrollbar=True
                        ):
                            dpg.add_spacer(height=15)

                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    " ANALISIS DATA MODEL", color=(26, 188, 156)
                                )
                                dpg.add_spacer(width=20)
                                dpg.add_button(
                                    label=" Render/Update Plot",
                                    callback=update_plot_callback,
                                )

                            dpg.add_separator()
                            dpg.add_spacer(height=10)

                            with dpg.group(horizontal=True):

                                with dpg.child_window(
                                    width=640,
                                    height=480,
                                    border=True,
                                    no_scrollbar=True,
                                ):
                                    dpg.add_spacer(height=10)
                                    with dpg.group(horizontal=True):
                                        dpg.add_spacer(width=20)
                                        dpg.add_image("plot_texture")

                                with dpg.child_window(
                                    width=-1, height=480, border=True, no_scrollbar=True
                                ):
                                    dpg.add_spacer(height=10)
                                    dpg.add_text(
                                        " RINGKASAN STATISTIK", color=(148, 163, 184)
                                    )
                                    dpg.add_separator()
                                    dpg.add_spacer(height=10)

                                    dpg.add_text(
                                        "Model: K-Nearest Neighbor",
                                        color=(230, 240, 235),
                                    )
                                    dpg.add_text(
                                        "Mean Akurasi: 85.2%", color=(74, 222, 128)
                                    )
                                    dpg.add_text(
                                        "Standar Deviasi: ± 5.0%", color=(250, 204, 21)
                                    )

                                    dpg.add_spacer(height=20)
                                    dpg.add_text(" EXPORT DATA", color=(148, 163, 184))
                                    dpg.add_separator()
                                    dpg.add_spacer(height=10)
                                    dpg.add_button(
                                        label=" Download Report (.CSV)",
                                        width=-1,
                                        height=30,
                                        callback=export_excel_report_callback 
                                    )
                                    
                                    dpg.add_spacer(height=5)
                                    dpg.add_text(" ", tag="export_status_msg", color=(148, 163, 184))

# ====================================
# =========== Styling Config =========
# ====================================
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(
            dpg.mvThemeCol_WindowBg, (28, 28, 28), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ChildBg, (40, 40, 40), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_Border, (70, 70, 70), category=dpg.mvThemeCat_Core
        )

        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (22, 160, 133), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (26, 188, 156), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (18, 130, 108), category=dpg.mvThemeCat_Core
        )

        dpg.add_theme_color(
            dpg.mvThemeCol_Header, (55, 55, 55), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_HeaderHovered, (75, 75, 75), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_HeaderActive, (26, 188, 156), category=dpg.mvThemeCat_Core
        )

        dpg.add_theme_color(
            dpg.mvThemeCol_PlotHistogram, (26, 188, 156), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_FrameBg, (20, 20, 20), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_FrameBgHovered, (45, 45, 45), category=dpg.mvThemeCat_Core
        )

        dpg.add_theme_color(
            dpg.mvThemeCol_Text, (230, 230, 230), category=dpg.mvThemeCat_Core
        )

        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)

        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

build_login_window()
build_main_windows()
build_register_window()


dpg.bind_theme(global_theme)
app_title = "Gestura"
dpg.create_viewport(
    title=app_title, width=1280, height=800, resizable=False, decorated=True
)

dpg.setup_dearpygui()

try:
    dpg.set_viewport_small_icon("src/gestura/assets/gestura-single-titlebar.ico")
    dpg.set_viewport_large_icon("src/gestura/assets/gestura-single-titlebar.ico")
except Exception as e:
    print(f"Warning: Failed to load custom icons. {e}")

dpg.show_viewport()
print("DEBUG: show_viewport done, is_running=", dpg.is_dearpygui_running())
dpg.set_primary_window("LoginWindow", True)
set_title_bar_color(app_title, 10, 10, 10)
print("DEBUG: before main loop, is_running=", dpg.is_dearpygui_running())

# ==========================================
# Variabel untuk loop kamera utama
# ==========================================
capture_cooldown = 0
prediction_buffer = deque(maxlen=15)
kalimat_terkumpul = ""
last_detected = ""

last_gesture_time = time.time()
AUTO_READ_TIMEOUT = 7.0



# ==========================================
# Main Loop gestur tangan dan kamera
# ==========================================
while dpg.is_dearpygui_running():
    if engine_running and cap is not None:
        ret, cap_frame = cap.read()
        if ret:
            cap_frame = cv2.flip(cap_frame, 1)

            points, hand_landmarks = get_hand_points_mediapipe(cap_frame)
            show_landmarks = dpg.get_value("show_lm_cb")

            if points is not None:
                if show_landmarks:
                    mp_draw.draw_landmarks(
                        cap_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    for i, (x, y) in enumerate(points):
                        cv2.circle(cap_frame, (int(x), int(y)), 4, (156, 188, 26), -1)
                        cv2.putText(
                            cap_frame,
                            str(i),
                            (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                for i in range(5):
                    lm = hand_landmarks.landmark[i]
                    dpg.set_value(f"t_x_{i}", f"{lm.x:.3f}")
                    dpg.set_value(f"t_y_{i}", f"{lm.y:.3f}")

                if capture_cooldown == 0:
                    A = points.flatten().reshape(1, -1)
                    hasil_huruf = gesture_engine.predict_gesture(A)

                    prediction_buffer.append(hasil_huruf)

                    if len(prediction_buffer) == 15:
                        huruf_terbanyak, jumlah_muncul = Counter(
                            prediction_buffer
                        ).most_common(1)[0]
                        
                        if jumlah_muncul >= 12:
                            
                            if huruf_terbanyak != last_detected:   
                                if huruf_terbanyak == "SPACE":
                                    kalimat_terkumpul += " "
                                    log_message(f"Teks: {kalimat_terkumpul}")
                                    
                                    last_gesture_time = time.time() 
                                else: 
                                    kalimat_terkumpul += huruf_terbanyak
                                    log_message(f"{kalimat_terkumpul}")    
                                    
                                    last_gesture_time = time.time()
                                    try:
                                        AudioPlayer.play_alphabet(huruf_terbanyak)
                                    except Exception as e:
                                        print(f"Gagal memutar audio: {e}")
                                
                                last_detected = huruf_terbanyak                            
                            
                            if dpg.does_item_exist("HurufPopup"):
                                dpg.delete_item("HurufPopup")

                            if huruf_terbanyak not in ["SPACE", "DONE"]:
                                with dpg.window(
                                    tag="HurufPopup",
                                    no_title_bar=True,
                                    pos=[300, 200],
                                    no_resize=True,
                                ):
                                    dpg.add_text(
                                        f"  {huruf_terbanyak}  ", color=(26, 188, 156) 
                                    )

                            prediction_buffer.clear()
                            capture_cooldown = 60
                            
            waktu_sekarang = time.time()
            
            if kalimat_terkumpul.strip() != "" and (waktu_sekarang - last_gesture_time) >= AUTO_READ_TIMEOUT:
                log_message(f"Auto-read: {kalimat_terkumpul}")
                threading.Thread(target=AudioPlayer._speak_task, args=(kalimat_terkumpul,), daemon=True).start()
                kalimat_terkumpul = ""
                last_detected = ""
                
                
            # ==========================================
            # Hasil Alfabet
            # ==========================================
            if capture_cooldown == 30:
                if dpg.does_item_exist("HurufPopup"):
                    dpg.delete_item("HurufPopup")

            if capture_cooldown > 0:
                capture_cooldown -= 1

            # ==========================================
            # Render frame kamera ke DPG
            # ==========================================
            rgba_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGBA)
            if rgba_frame.shape[1] != cam_width or rgba_frame.shape[0] != cam_height:
                rgba_frame = cv2.resize(rgba_frame, (cam_width, cam_height))

            texture_data = rgba_frame.astype(np.float32) / 255.0
            dpg.set_value("camera_texture", texture_data.flatten())

    dpg.render_dearpygui_frame()

if cap is not None:
    cap.release()
hands.close()
dpg.destroy_context()
