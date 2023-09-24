import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

model = tf.keras.models.load_model('image_enhancement_model.keras')

input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width * 4, frame_height * 4))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def enhance_image(image):
    input_image = cv2.resize(image, (214, 144))

    input_image = input_image.astype(np.float32) / 255.0

    enhanced_image = model.predict(np.expand_dims(input_image, axis=0))[0]

    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image


for _ in tqdm(range(total_frames), ncols=100):
    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = enhance_image(frame)

    enhanced_frame = cv2.resize(enhanced_frame, (frame_width * 4, frame_height * 4))

    output.write(enhanced_frame)

cap.release()
output.release()
cv2.destroyAllWindows()
