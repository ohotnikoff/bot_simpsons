from flask import Flask
from pathlib import Path
import numpy as np
import torch
import pickle

import simplecnn
import dataset


app = Flask(__name__)

# работаем на видеокарте
DEVICE = torch.device("cpu")


def predict_one_sample(model, inputs, device=DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs


@app.route('/')
def predict():
    # random_characters = int(np.random.uniform(0, 1000))
    random_characters = int(0)

    val_files = []  # posix | from pathlib import Path
    # file = Path('files/pic_0610.jpg')
    # file = Path('files/test_1.jpg')
    # file = Path('files/test_2.jpg')
    file = Path('files/test_5.jpg')
    val_files.append(file)
    val_dataset = dataset.SimpsonsDataset(val_files, mode='val')
    ex_img, true_label = val_dataset[random_characters]

    # predict
    probs_im = predict_one_sample(model, ex_img.unsqueeze(0))
    y_pred = np.argmax(probs_im)
    print(y_pred)
    predicted_label = label_encoder.classes_[y_pred]
    print(predicted_label)
    accuracy = np.max(probs_im) * 100

    output = "{} с точностью: {:.0f}%".format(predicted_label, accuracy)
    return output


if __name__ == '__main__':
    SimpleCnn = simplecnn.SimpleCnn

    n_classes = 42
    model = SimpleCnn(42).to(DEVICE)
    print(torch.__version__)
    with open('torch_simple_cnn.pkl', 'rb') as f:
        model.load_state_dict(torch.load(
            f,
            map_location=DEVICE
        ))
    model.to(DEVICE)

    # label_encoder = LabelEncoder()
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    app.run(debug=True)
