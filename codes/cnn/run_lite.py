import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\admin\Desktop\L.D.D\codes\cnn\savedModels\leaf_badam_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (replace with your actual classes)
class_names = ['good', 'bad']

def preprocess_frame(frame):
    # Resize to model input size (64x64) and normalize to [0,1]
    img = cv2.resize(frame, (64, 64))
    img = img.astype('float32') / 255.0
    # Add batch dimension and convert to float32
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    return input_data

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB if your model expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess frame
    input_data = preprocess_frame(rgb_frame)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output predictions
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get predicted class and confidence
    class_idx = np.argmax(output_data)
    confidence = output_data[class_idx]

    label = f"{class_names[class_idx]}: {confidence*100:.2f}%"

    # Display label on original frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show webcam frame with prediction
    cv2.imshow('Real-time Leaf Disease Detection', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
