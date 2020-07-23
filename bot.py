import config
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id

from pathlib import Path
import numpy as np
import torch
import simplecnn
import dataset
import pickle
import requests


def save_image(image_url):
    filename = 'files/' + image_url.split('/')[-1]
    with open(filename, 'wb') as handle:
        response = requests.get(image_url, stream=True)
        if not response.ok:
            print(response)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    return filename


def prepare_image(image):
    file = Path(image)
    val_files = []
    val_files.append(file)
    val_dataset = dataset.SimpsonsDataset(val_files, mode='val')
    return val_dataset[0]


def load_model(n_classes):
    SimpleCnn = simplecnn.SimpleCnn
    model = SimpleCnn(n_classes).to(config.DEVICE)
    with open(config.FILE_MODEL, 'rb') as f:
        model.load_state_dict(torch.load(
            f,
            map_location=config.DEVICE
        ))
    model.to(config.DEVICE)
    return model


def load_label_encoder():
    with open(config.FILE_LABEL_ENCODER, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


def predict_one_sample(model, inputs, device=config.DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs


def predict(image):
    ex_img, true_label = prepare_image(image)

    # load model
    model = load_model(42)
    # load label_encoder
    label_encoder = load_label_encoder()

    # predict
    probs_im = predict_one_sample(model, ex_img.unsqueeze(0))
    y_pred = np.argmax(probs_im)
    predicted_label = label_encoder.classes_[y_pred]
    accuracy = np.max(probs_im) * 100

    output = "{} с точностью: {:.0f}%".format(predicted_label, accuracy)
    return output


def write_msg(vk_session, user_id, message):
    vk_session.method('messages.send', {
        'user_id': user_id,
        'random_id': get_random_id(),
        'message': message
    })


def main():
    # Авторизуемся как сообщество
    vk_session = vk_api.VkApi(token=config.VK_API_TOKEN)

    # Работа с сообщениями
    longpoll = VkBotLongPoll(vk_session, config.GROUP_ID)

    # Основной цикл
    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:

            # Получаем изображения
            if event.obj.attachments:
                for attach in event.obj.attachments:
                    if attach['type'] != 'photo':
                        continue
                    for photo in attach['photo']['sizes']:
                        if photo['type'] == 'x':
                            print(photo['url'])
                            image = save_image(photo['url'])
                            result = predict(image)
                            print(result)
                            write_msg(vk_session, event.obj.from_id, result)


if __name__ == '__main__':
    main()
