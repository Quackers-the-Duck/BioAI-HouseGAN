import tensorflow as tf
import matplotlib.pyplot as plt

# The path to the .h5 checkpoint that was saved after the training loop:
MODEL_PATH = '/Users/Amy/Documents/GitHub/AI-Queens/Pix2Pix Trained Models/generator_model_001_house2plan_25epochs.h5'

# Location of the image you want to test with. The images should be an RGB .png or .jpg file, scaled to 256x256 pixels
INPUT_IMAGE_PATH = '/Users/Amy/Documents/GitHub/AI-Queens/Data/Augmented/A/ff4469a26857.png'

model = tf.keras.models.load_model(MODEL_PATH)

img = tf.io.read_file(INPUT_IMAGE_PATH)
img = tf.io.decode_png(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, (256, 256), antialias=True)
img.shape

plt.imshow(img)
plt.axis('off')
plt.show()

input_image = tf.expand_dims(img, axis=[0])

# prediction = model.predict(input_image)
prediction = model(input_image, training=True)

plt.imshow(prediction[0, :, :, :])
plt.axis('off')
plt.show()