import PySimpleGUI as sg
import cv2
from easyocr import Reader
from PIL import Image
import io
from deep_translator import GoogleTranslator
import language_tool_python


def prepare_image(path):
    res = Image.open(path)
    res.thumbnail((400, 400))
    bio = io.BytesIO()
    res.save(bio, format="PNG")
    return bio.getvalue()


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    return "".join([c if ord(c) < 128 or (ord(c) >= 1040 and ord(c) <= 1103) else "" for c in text]).strip()


def text_to_send(text):
    s = ''
    for t in text:
        s += t + '\n'
    return s


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    return blurred


def process(path, lang, gpu):
    print(f'OCR`ing with the following language: {lang}')
    image = preprocess_image(path)

    reader = Reader([lang], gpu=gpu > 0)
    results = reader.readtext(image)

    all_text = []

    for (bbox, text, prob) in results:
        all_text.append(text)

        # Unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # Cleanup the text and draw the box surrounding the text
        text = cleanup_text(text)
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output image
    cv2.imwrite('res.png', image)
    result = prepare_image('res.png')
    res_text = text_to_send(all_text)
    return result, res_text


def create_help_window():
    layout_help = [
        [sg.Text('Нажмите на кнопку "Выберите изображение", чтобы перейти к поиску необходимой фотографии.')],
        [sg.Text('Щелкните на выпадающий список, чтобы выбрать язык текста, который необходимо распознать на изображении:')],
        [sg.Text('Поддерживаемые языки: Русский, Английский, Испанский, Немецкий, Французский, Кыргызский, Итальянский, Португальский, Голландский, Польский, Арабский, Японский, Китайский (упрощённый), Корейский, Хинди')],
        [sg.Text('Нажмите на кнопку "Начать" и ожидайте окончания выполнения процесса.')],
        [sg.Button('Закрыть')]
    ]
    return sg.Window('Помощь', layout_help)


# Setup GUI
layout = [
    [sg.Text('Пожалуйста, выберите изображение:')],
    [sg.InputText(key='-FILEPATH-'), sg.FileBrowse(button_text='Выбрать файл...'),
     sg.FileBrowse(file_types=[('JPG Files', '*.jpg'), ('PNG Files', '*.png'), ('JPEG Files', '*.jpeg')])],
    [sg.Text('Пожалуйста, выберите язык текста:')],
    [sg.Combo(['Русский', 'Английский', 'Испанский', 'Немецкий', 'Французский', 'Кыргызский', 'Итальянский', 'Португальский', 'Голландский', 'Польский', 'Арабский', 'Японский', 'Китайский (упрощённый)', 'Корейский', 'Хинди'],
              key='-COMBO-', default_value='Русский')],
    [sg.Text('Пожалуйста, выберите язык для перевода:')],
    [sg.Combo(['Русский', 'Английский', 'Испанский', 'Немецкий', 'Французский', 'Кыргызский', 'Итальянский', 'Португальский', 'Голландский', 'Польский', 'Арабский', 'Японский', 'Китайский (упрощённый)', 'Корейский', 'Хинди'],
              key='-TRANSLATE-', default_value='Русский')],
    [sg.Text('▶️ текст с картинки')],
    [sg.Multiline(key='-TEXT-', size=(50, 5),
                  default_text='Здесь будет распознанный текст...')],
    [sg.Text('🔄 текст, который был переведен')],
    [sg.Multiline(key='-TRANSLATED-', size=(50, 5),
                  default_text='Здесь будет переведённый текст...')],
    [sg.Image(key='-IMAGE-')],
    [sg.Button('Начать'), sg.Button('Выход'), sg.Push(), sg.Button('Помощь')]
]

window = sg.Window('Распознавание рукописного текста и перевод', layout)

# Словарь для преобразования полного названия языка в сокращение для OCR и перевода
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

# Создаём экземпляр инструмента для проверки грамматики
tool = language_tool_python.LanguageTool('ru')  # Используем русский язык для примера

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Выход':
        break

    if event == 'Начать':
        # Получаем язык для перевода
        translate_lang = language_dict[values['-TRANSLATE-']]

        r, t = process(values['-FILEPATH-'], language_dict[values['-COMBO-']], -1)

        # Проверяем орфографию и грамматику в оригинальном тексте
        matches = tool.check(t)
        corrected_text = language_tool_python.utils.correct(t, matches)

        # Переводим текст на выбранный язык
        try:
            translated_text = GoogleTranslator(source='auto', target=translate_lang).translate(corrected_text)
        except Exception as e:
            translated_text = f"Ошибка перевода: {str(e)}"

        # Обновляем GUI
        window['-IMAGE-'].update(data=r)
        window['-TEXT-'].update(t)  # Оригинальный текст с картинки
        window['-TRANSLATED-'].update(translated_text)  # Переведённый текст
        print(translated_text)

    if event == 'Помощь':
        window_help = create_help_window()
        while True:
            event2, values2 = window_help.read()
            if event2 == 'Закрыть' or event2 == sg.WINDOW_CLOSED:
                break
        window_help.close()

window.close()
