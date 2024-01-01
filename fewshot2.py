import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your data directory
data_dir = "/Users/satvikrajselar/Desktop/MLprojectwork/flowers"

# Specify the image size and batch size
img_size = (224, 224)
batch_size = 32

# Create an ImageDataGenerator with optional data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Use the flow_from_directory method to load images from the specified directory
data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # 'categorical' for multi-class classification
    shuffle=True
)

# Load EfficientNetB0 model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classifier on top of EfficientNetB0
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # Adjust the number of output classes

model_efficientnet = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model_efficientnet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_efficientnet = model_efficientnet.fit(
    data_generator,
    epochs=10,  # Adjust the number of epochs
    steps_per_epoch=len(data_generator),
    validation_data=data_generator,
    validation_steps=len(data_generator)
)

# Evaluate the model
test_loss_efficientnet, test_acc_efficientnet = model_efficientnet.evaluate(data_generator, steps=len(data_generator))
print(f"Test Loss: {test_loss_efficientnet}, Test Accuracy: {test_acc_efficientnet}")