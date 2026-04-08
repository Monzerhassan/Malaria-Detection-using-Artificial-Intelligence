
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk

model = tf.keras.models.load_model("model/malaria_model.h5")

def predict_image():
    file_path = filedialog.askopenfilename()

    img = Image.open(file_path).resize((200,200))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = f"❌ مصاب بالملاريا\n{prediction*100:.2f}%"
    else:
        result = f"✅ غير مصاب\n{(1-prediction)*100:.2f}%"

    label.config(text=result)


root = tk.Tk()
root.title("Malaria Detection System")
root.geometry("400x400")

btn = tk.Button(root, text="اختر صورة", command=predict_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

label = tk.Label(root, text="", font=("Arial", 14))
label.pack(pady=10)

root.mainloop()
