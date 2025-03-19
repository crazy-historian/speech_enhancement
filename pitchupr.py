import tkinter as tk
from tkinter import messagebox
import parselmouth
import numpy as np
import time
import threading
import sys

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# ---------------- ПАРАМЕТРЫ АУДИООБРАБОТКИ ----------------
METHOD = "ac"
PITCH_FLOOR = 100
PITCH_CEILING = 600
TIME_STEP = 0.01
SILENCE_THRESHOLD = 0.03
VOICING_THRESHOLD = 0.45
OCTAVE_COST = 0.01
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14
BLOCKSIZE = 1024

# ---------------- ЧТЕНИЕ ТЕКСТА ----------------
def load_exercises_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        exercises = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
        return exercises
    except FileNotFoundError:
        return []

# ---------------- ФУНКЦИЯ ЦВЕТА ДЛЯ PITCH ----------------
def get_color_for_pitch(pitch, min_freq, max_freq):
    if pitch == 0:
        return "#808080"  # серый
    return "#00FF00" if min_freq <= pitch <= max_freq else "#FF0000"  # зелёный или красный


class PitchTrainerApp:
    def __init__(self, root, exercises):
        self.root = root
        self.exercises = exercises
        self.current_ex_index = 0

        self.min_freq = 0
        self.max_freq = 0
        self.is_range_set = False

        self.recording = False
        self.audio_thread = None
        self.pitch_values = []
        self.total_count = 0
        self.in_range_count = 0

        self.init_ui()

    def init_ui(self):
        self.root.title("Pitch Trainer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(bg="#303030")

        self.freq_label = tk.Label(self.root, text="Текущая частота: 0 Гц",
                                   font=("Arial", 14), bg="#4A4A4A", fg="white")
        self.freq_label.pack(pady=5)

        self.range_label = tk.Label(self.root, text="Диапазон не задан",
                                    font=("Arial", 12), bg="#4A4A4A", fg="white")
        self.range_label.pack(pady=5)

        self.exercise_label = tk.Label(self.root, text="", font=("Arial", 16),
                                       wraplength=600, bg="#5A5A5A", fg="white")
        self.exercise_label.pack(pady=10)

        self.button_frame = tk.Frame(self.root, bg="#303030")
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        btn_style = {"fg": "black", "bg": "#B0B0B0", "font": ("Arial", 11)}

        self.set_range_button = tk.Button(self.button_frame, text="Задать диапазон",
                                          command=self.set_range, **btn_style)
        self.set_range_button.pack(side=tk.LEFT, padx=5)

        self.start_button = tk.Button(self.button_frame, text="Начать",
                                      command=self.start_recording, **btn_style)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.finish_button = tk.Button(self.button_frame, text="Закончить",
                                       command=self.finish_recording, state=tk.DISABLED, **btn_style)
        self.finish_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.root, text="Перейти к следующему упражнению",
                                     command=self.next_exercise, **btn_style)
        self.next_button.pack_forget()

        self.show_exercise(0)

    def show_exercise(self, index):
        if index < len(self.exercises):
            self.exercise_label.config(text=self.exercises[index])
            self.set_range_button.config(state=tk.NORMAL if index == 0 else tk.DISABLED)
        else:
            self.show_final_result()

    def set_range(self):
        range_win = tk.Toplevel(self.root)
        range_win.title("Целевой диапазон")
        range_win.geometry("300x150")
        range_win.configure(bg="#303030")

        label_style = {"bg": "#303030", "fg": "white", "font": ("Arial", 11)}

        tk.Label(range_win, text="Нижняя граница (Гц):", **label_style).pack(pady=2)
        min_entry = tk.Entry(range_win, fg="white", bg="#4A4A4A", bd=2)
        min_entry.pack(pady=5)

        tk.Label(range_win, text="Верхняя граница (Гц):", **label_style).pack(pady=2)
        max_entry = tk.Entry(range_win, fg="white", bg="#4A4A4A", bd=2)
        max_entry.pack(pady=5)

        def apply_range():
            try:
                mn, mx = float(min_entry.get()), float(max_entry.get())
                if mn < 0 or mx <= mn:
                    raise ValueError
                self.min_freq, self.max_freq = mn, mx
                self.is_range_set = True
                self.range_label.config(text=f"Диапазон: {mn:.1f} - {mx:.1f} Гц")
                range_win.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Неверный ввод диапазона!")

        tk.Button(range_win, text="OK", command=apply_range, fg="black", bg="#B0B0B0").pack(pady=10)

    def start_recording(self):
        if not self.is_range_set and self.current_ex_index == 0:
            messagebox.showinfo("Инфо", "Сначала задайте целевой диапазон!")
            return

        self.recording = True
        self.pitch_values.clear()
        self.total_count = 0
        self.in_range_count = 0

        self.start_button.config(state=tk.DISABLED)
        self.finish_button.config(state=tk.NORMAL)

        self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.audio_thread.start()

    def finish_recording(self):
        self.recording = False
        self.finish_button.config(state=tk.DISABLED)

        if self.audio_thread and self.audio_thread.is_alive():
            time.sleep(0.1)
            self.audio_thread.join()

        avg_pitch = sum(self.pitch_values) / len(self.pitch_values) if self.pitch_values else 0
        color = get_color_for_pitch(avg_pitch, self.min_freq, self.max_freq)
        percent_in_range = (self.in_range_count / self.total_count) * 100 if self.total_count else 0

        result_label = tk.Label(self.root, text=f"Средняя частота: {avg_pitch:.1f} Гц\n"
                                                f"Процент попадания: {percent_in_range:.1f}%",
                                font=("Arial", 14), bg="#303030", fg=color)
        result_label.pack()

        if self.current_ex_index < len(self.exercises) - 1:
            self.next_button.pack(pady=10)
        else:
            tk.Button(self.root, text="Выйти", command=self.on_close, bg="#B0B0B0", fg="black").pack(pady=10)

    def next_exercise(self):
        self.next_button.pack_forget()
        self.current_ex_index += 1
        for widget in self.root.pack_slaves():
            if isinstance(widget, tk.Label) and widget not in (self.freq_label, self.range_label, self.exercise_label):
                widget.destroy()
        self.show_exercise(self.current_ex_index)
        self.start_button.config(state=tk.NORMAL)

    def show_final_result(self):
        tk.Label(self.root, text="Все упражнения выполнены!", font=("Arial", 16), fg="white", bg="#303030").pack(pady=10)
        tk.Button(self.root, text="Выйти", command=self.on_close, bg="#B0B0B0", fg="black").pack(pady=10)

    def on_close(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    exercises = load_exercises_from_file("example.txt")
    if exercises:
        root = tk.Tk()
        app = PitchTrainerApp(root, exercises)
        root.mainloop()
