import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/malaria_model.h5")

img_path = input("ادخل مسار الصورة: ")

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"❌ مصاب بنسبة {prediction*100:.2f}%")
else:
    print(f"✅ غير مصاب بنسبة {(1-prediction)*100:.2f}%")
