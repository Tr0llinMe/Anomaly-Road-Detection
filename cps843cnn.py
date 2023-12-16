import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
train_dir = 'C:/Users/huyda/Desktop/CPS843_Final/train'
validation_dir = 'C:/Users/huyda/Desktop/CPS843_Final/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
print(f"Found {train_generator.samples} images in training directory.")

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
print(f"Found {validation_generator.samples} images in validation directory.")

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=3,  # depends on your data - was 100
    epochs=15,
    validation_data=validation_generator,
    validation_steps=3)  # depends on your data - was 50

# Function to predict and return confidence score along with the file name
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    label = 'Abnormal' if confidence > 0.5 else 'Normal'
    return image_path, label, confidence

# Example usage
file_path, label, confidence = predict_image('C:/Users/huyda/Desktop/CPS843_Final/test/test6.jpg')
print(f"File: {file_path}, Prediction: {label}, Confidence: {confidence}")
