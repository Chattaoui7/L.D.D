import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing import image

# Build your model architecture (same as training)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),input_shape=(64,64,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3,activation='softmax'))

# Load the trained weights
model.load_weights(r'C:\Users\admin\Desktop\Leaf-Disease-Detection\codes\cnn\weights.best.from_scratch.hdf5')

# Class labels — replace with your actual class names
class_names = ['blight', 'mildew', 'rust']

def preprocess_frame(frame):
    # Resize frame to 64x64 like in training
    img = cv2.resize(frame, (64, 64))
    # Convert to array and normalize
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR(OpenCV default) to RGB (Keras uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    input_tensor = preprocess_frame(rgb_frame)

    # Predict
    preds = model.predict(input_tensor)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    label = f"{class_names[class_idx]}: {confidence*100:.2f}%"

    # Display label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Real-time Detection', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
