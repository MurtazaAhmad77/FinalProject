import PySimpleGUI as sg
from easyocr import Reader
from deep_translator import GoogleTranslator
import language_tool_python
import cv2
from PIL import Image
import io
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


# Функция обработки изображения
def prepare_image(path):
    res = Image.open(path)
    res.thumbnail((400, 400))
    bio = io.BytesIO()
    res.save(bio, format="PNG")
    return bio.getvalue()

def cleanup_text(text):
    return "".join([c if ord(c) < 128 or (ord(c) >= 1040 and ord(c) <= 1103) else "" for c in text]).strip()

def text_to_send(text):
    return '\n'.join(text)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    return blurred

def process(path, lang, gpu):
    print(f'OCRing with the following language: {lang}')
    image = preprocess_image(path)

    reader = Reader([lang], gpu=gpu > 0)
    results = reader.readtext(image)

    all_text = []

    for (bbox, text, prob) in results:
        all_text.append(text)

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        text = cleanup_text(text)
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite('res.png', image)
    result = prepare_image('res.png')
    res_text = text_to_send(all_text)
    return result, res_text

def create_help_window():
    layout_help = [
        [sg.Text('🔹 Как пользоваться программой:\n')],
        [sg.Text(' Нажмите кнопку "Выбрать файл...", чтобы загрузить изображение.')],
        [sg.Text(' Выберите язык, на котором написан текст на изображении.')],
        [sg.Text(' Выберите язык, на который вы хотите перевести текст.')],
        [sg.Text(' Нажмите "Начать" и дождитесь завершения обработки.\n')],
        [sg.Text('🔹 Поддерживаемые языки:')],
        [sg.Text(', '.join(language_dict.keys()))],
        [sg.Text('\n🔹 Поддерживаемые форматы изображений:')],
        [sg.Text(' JPG, JPEG, PNG')],
        [sg.Text(' Максимальный размер файла: 5 МБ')],
        [sg.Text(' Файл не должен быть пустым.')],
        [sg.Text(' Язык текста и язык перевода должны быть разными.')],
        [sg.Button('Закрыть')]
    ]
    return sg.Window('Помощь', layout_help)


def show_error(message):
    layout = [[sg.Text(message)], [sg.Button('Закрыть')]]
    window = sg.Window('Ошибка', layout, modal=True)
    while True:
        event, _ = window.read()
        if event == 'Закрыть' or event == sg.WINDOW_CLOSED:
            break
    window.close()

# GUI
layout = [
    [sg.Text('Пожалуйста, выберите изображение:')],
    [sg.InputText(key='-FILEPATH-'), sg.FileBrowse(button_text='Выбрать файл...', file_types=[('Image Files', '*.jpg;*.jpeg;*.png')])],
    [sg.Text('Пожалуйста, выберите язык текста:')],
    [sg.Combo(['Русский', 'Английский', 'Испанский', 'Немецкий', 'Французский', 'Кыргызский', 'Итальянский', 'Португальский', 'Голландский', 'Польский', 'Арабский', 'Японский', 'Китайский (упрощённый)', 'Корейский', 'Хинди'],
              key='-COMBO-', default_value='Русский')],
    [sg.Text('Пожалуйста, выберите язык для перевода:')],
    [sg.Combo(['Русский', 'Английский', 'Испанский', 'Немецкий', 'Французский', 'Кыргызский', 'Итальянский', 'Португальский', 'Голландский', 'Польский', 'Арабский', 'Японский', 'Китайский (упрощённый)', 'Корейский', 'Хинди'],
              key='-TRANSLATE-', default_value='Русский')],
    [sg.Text('▶️ текст с картинки')],
    [sg.Multiline(key='-TEXT-', size=(50, 5), default_text='Здесь будет распознанный текст...')],
    [sg.Text('🔄 текст, который был переведен')],
    [sg.Multiline(key='-TRANSLATED-', size=(50, 5), default_text='Здесь будет переведённый текст...')],
    [sg.Image(key='-IMAGE-')],
    [sg.Button('Начать'), sg.Button('Сбросить'), sg.Button('Выход'), sg.Push(), sg.Button('Помощь')]
]

window = sg.Window('Распознавание рукописного текста и перевод', layout, finalize=True)

# Привязка горячих клавиш к Multiline-полям
def bind_multiline_hotkeys(window, keys):
    for key in keys:
        widget = window[key].Widget
        widget.bind('<Control-a>', lambda e, w=widget: (w.tag_add('sel', '1.0', 'end'), 'break'))
        widget.bind('<Control-A>', lambda e, w=widget: (w.tag_add('sel', '1.0', 'end'), 'break'))
        widget.bind('<Control-c>', lambda e, w=widget: (w.event_generate("<<Copy>>"), 'break'))
        widget.bind('<Control-C>', lambda e, w=widget: (w.event_generate("<<Copy>>"), 'break'))
        widget.bind('<Control-v>', lambda e, w=widget: (w.event_generate("<<Paste>>"), 'break'))
        widget.bind('<Control-V>', lambda e, w=widget: (w.event_generate("<<Paste>>"), 'break'))
        widget.bind('<Control-x>', lambda e, w=widget: (w.event_generate("<<Cut>>"), 'break'))
        widget.bind('<Control-X>', lambda e, w=widget: (w.event_generate("<<Cut>>"), 'break'))

bind_multiline_hotkeys(window, ['-TEXT-', '-TRANSLATED-'])


language_dict = {
    'Русский': 'ru',
    'Английский': 'en',
    'Испанский': 'es',
    'Немецкий': 'de',
    'Французский': 'fr',
    'Кыргызский': 'ky',
    'Итальянский': 'it',
    'Португальский': 'pt',
    'Голландский': 'nl',
    'Польский': 'pl',
    'Арабский': 'ar',
    'Японский': 'ja',
    'Китайский (упрощённый)': 'zh-cn',
    'Корейский': 'ko',
    'Хинди': 'hi'
}

tool = language_tool_python.LanguageTool('ru')

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Выход':
        break

    if event == 'Начать':
        if not values['-FILEPATH-']:
            show_error('Пожалуйста, выберите изображение.')
            continue
        if not values['-COMBO-']:
            show_error('Пожалуйста, выберите язык текста.')
            continue
        if not values['-TRANSLATE-']:
            show_error('Пожалуйста, выберите язык для перевода.')
            continue
        if not os.path.isfile(values['-FILEPATH-']):
            show_error('Файл не найден. Пожалуйста, выберите другой файл.')
            continue
        if not values['-FILEPATH-'].endswith(('.jpg', '.jpeg', '.png')):
            show_error('Неподдерживаемый формат файла. Пожалуйста, выберите изображение в формате JPG или PNG.')
            continue
        if values['-COMBO-'] == values['-TRANSLATE-']:
            show_error('Язык текста и язык перевода не должны совпадать.')
            continue

        
        file_size = os.path.getsize(values['-FILEPATH-'])
        if file_size > 5 * 1024 * 1024:
            show_error('Файл слишком большой. Пожалуйста, выберите файл размером не более 5 МБ.')
            continue
        if file_size == 0:
            show_error('Файл пустой. Пожалуйста, выберите другой файл.')
            continue

        image_path = values['-FILEPATH-']
        ocr_lang = language_dict[values['-COMBO-']]
        translate_lang = language_dict[values['-TRANSLATE-']]

        try:
            r, t = process(image_path, ocr_lang, -1)
        except Exception as e:
            show_error(f'Ошибка обработки изображения: {str(e)}')
            continue


        matches = tool.check(t)
        corrected_text = language_tool_python.utils.correct(t, matches)

        try:
            translated_text = GoogleTranslator(source='auto', target=translate_lang).translate(corrected_text)
        except Exception as e:
            translated_text = f"Ошибка перевода: {str(e)}"

        window['-IMAGE-'].update(data=r)
        window['-TEXT-'].update(t)
        window['-TRANSLATED-'].update(translated_text)

    if event == 'Сбросить':
        window['-FILEPATH-'].update('')
        # Оставляем язык и язык перевода выбранными
        window['-TEXT-'].update('')
        window['-TRANSLATED-'].update('')
        window['-IMAGE-'].update(data=None)

    if event == 'Помощь':
        window_help = create_help_window()
        while True:
            event2, _ = window_help.read()
            if event2 == 'Закрыть' or event2 == sg.WINDOW_CLOSED:
                break
        window_help.close()

window.close()
