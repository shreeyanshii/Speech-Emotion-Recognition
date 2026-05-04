import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import traceback
import tensorflow as tf

try:
    m = tf.keras.models.load_model('Deep Learning/SER_model.h5')
    print("Loaded successfully")
except Exception as e:
    with open('error.txt', 'w') as f:
        f.write(traceback.format_exc())
    print("Failed to load")
