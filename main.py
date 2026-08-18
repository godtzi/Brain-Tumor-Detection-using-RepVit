import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = r"models\repvit_brain_tumor.onnx"
IMAGE_PATH = "test2.jpg"


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


image = cv2.imread(IMAGE_PATH)
image_resized = cv2.resize(image, (224, 224))
image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
image_resized = image_resized.astype(np.float32) / 255.0
image_resized = np.transpose(image_resized, (2, 0, 1))
image_resized = np.expand_dims(image_resized, axis=0)


output = session.run(None, {
    input_name: image_resized
})

print("Output model:")
print(output)

prediction = np.argmax(output[0], axis=1)[0]

print("Predicted class:", prediction)