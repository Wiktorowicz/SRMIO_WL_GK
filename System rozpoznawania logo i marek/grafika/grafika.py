import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "logotypyMarek.keras"
LABELS_PATH = "labelsy.json"
IMAGE_SIZE = (224, 224)

try:
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
except Exception as e:
    messagebox.showerror("Blad", f"Nie udalo sie wczytac modelu lub etykiet.\n{e}")
    raise SystemExit()

root = tk.Tk()
root.title("System Rozpoznawania Logotypow")
root.geometry("900x800")
root.resizable(False, False)
root.configure(bg="#f4f4f4")

loaded_image = None
image_panel = None
camera_running = False
cap = None
camera_panel = None

def preprocess_for_model(img):
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(img.astype("float32"), axis=0)
    arr = preprocess_input(arr)
    return arr

def predict_from_array(img):
    arr = preprocess_for_model(img)
    preds = model.predict(arr, verbose=0)[0]
    top3 = preds.argsort()[-3:][::-1]
    return [(labels[i], preds[i] * 100) for i in top3]

def load_image():
    global loaded_image, image_panel
    file_path = filedialog.askopenfilename(
        filetypes=[("Obrazy", "*.jpg;*.jpeg;*.png;*.webp")]
    )
    if not file_path:
        return
    img = Image.open(file_path).convert("RGB")
    loaded_image = np.array(img)
    preview = img.resize((450, 450))
    preview_tk = ImageTk.PhotoImage(preview)
    if image_panel is None:
        image_panel = tk.Label(root, image=preview_tk, bg="#f4f4f4")
        image_panel.image = preview_tk
        image_panel.pack(pady=20)
    else:
        image_panel.config(image=preview_tk)
        image_panel.image = preview_tk
    result_label.config(text="Obraz zaladowany. Kliknij Rozpoznaj.", fg="#333")

def predict_file():
    if loaded_image is None:
        messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
        return
    top3 = predict_from_array(loaded_image)
    text = "TOP 3 wyniki:\n\n"
    for label, prob in top3:
        text += f"{label}: {prob:.2f}%\n"
    result_label.config(text=text, fg="#222")

def update_camera():
    global cap, camera_running, camera_panel
    if not camera_running:
        return
    ret, frame = cap.read()
    if not ret:
        return
    top3 = predict_from_array(frame)
    best_label, best_prob = top3[0]
    cv2.putText(
        frame, f"{best_label} ({best_prob:.1f}%)",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(img)
    if camera_panel is None:
        camera_panel = tk.Label(root, image=imgtk, bg="#f4f4f4")
        camera_panel.image = imgtk
        camera_panel.pack()
    else:
        camera_panel.config(image=imgtk)
        camera_panel.image = imgtk
    root.after(30, update_camera)

def start_camera():
    global cap, camera_running
    if camera_running:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Blad", "Nie udalo sie uruchomic kamery.")
        return
    camera_running = True
    result_label.config(text="Kamera uruchomiona.", fg="#333")
    update_camera()

def clear_screen():
    global loaded_image, image_panel, camera_panel, camera_running, cap
    camera_running = False
    if cap is not None:
        cap.release()
        cap = None
    if camera_panel is not None:
        camera_panel.destroy()
        camera_panel = None
    if image_panel is not None:
        image_panel.destroy()
        image_panel = None
    loaded_image = None
    result_label.config(
        text="Wczytaj obraz lub uruchom kamere.",
        fg="#555"
    )

title = tk.Label(
    root,
    text="System Rozpoznawania Logotypow",
    font=("Arial", 22, "bold"),
    bg="#f4f4f4",
    fg="#222"
)
title.pack(pady=10)

button_frame = tk.Frame(root, bg="#f4f4f4")
button_frame.pack(pady=10)

btn_load = tk.Button(
    button_frame, text="Wczytaj obraz",
    command=load_image, width=18, height=2,
    bg="#4CAF50", fg="white", font=("Arial", 12, "bold")
)
btn_load.grid(row=0, column=0, padx=10)

btn_predict = tk.Button(
    button_frame, text="Rozpoznaj",
    command=predict_file, width=18, height=2,
    bg="#2196F3", fg="white", font=("Arial", 12, "bold")
)
btn_predict.grid(row=0, column=1, padx=10)

btn_cam_on = tk.Button(
    button_frame, text="Uruchom kamere",
    command=start_camera, width=18, height=2,
    bg="#FF9800", fg="white", font=("Arial", 12, "bold")
)
btn_cam_on.grid(row=0, column=2, padx=10)

btn_clear = tk.Button(
    button_frame, text="Wyczysc ekran",
    command=clear_screen, width=18, height=2,
    bg="#9E9E9E", fg="white", font=("Arial", 12, "bold")
)
btn_clear.grid(row=0, column=3, padx=10)

result_label = tk.Label(
    root,
    text="Wczytaj obraz lub uruchom kamere.",
    font=("Arial", 14),
    bg="#f4f4f4",
    fg="#555"
)
result_label.pack(pady=20)

root.mainloop()
