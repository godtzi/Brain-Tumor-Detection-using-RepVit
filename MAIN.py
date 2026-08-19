import cv2
import numpy as np
import onnxruntime as ort
import torch



model_onnx = "models/repvit_brain_tumor"
model_onnx_int8 = "models/repvit_brain_tumor_int8"
model_pytorch = "models/repvit_brain_tumor.pt"

default_mode = "onnx"

IMAGE_PATH = "test_image/test1.jpg"

def mode_onnx():
    session = ort.InferenceSession(
        model_onnx,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    output = session.run(None, {
        input_name: image
        }
    )

    prediction = np.argmax(output[0], axis=1)[0]
    return prediction

def mode_pt():
    model = torch.jit.load(model_pytorch)
    model.eval()

    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image)

    with torch.no_grad():
        output = model(image)

    prediction = torch.argmax(output, dim=1).item()

    return prediction

def mode_int8():
    session = ort.InferenceSession(
        model_onnx_int8,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    output = session.run(None, {
        input_name: image
    })

    prediction = np.argmax(output[0], axis=1)[0]

    return prediction


print("Start")  

print(mode_int8())




