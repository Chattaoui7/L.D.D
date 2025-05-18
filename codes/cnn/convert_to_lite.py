import tensorflow as tf

#Load .keras or .h5
model = tf.keras.models.load_model(r"C:\Users\admin\Desktop\L.D.D\codes\cnn\savedModels\leaf_badam_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: optimize for size/speed
tflite_model = converter.convert()

# Save .tflite file
with open(r"C:\Users\admin\Desktop\L.D.D\codes\cnn\savedModels\leaf_badam_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as leaf_model.tflite")
