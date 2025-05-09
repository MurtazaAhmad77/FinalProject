# Распознавание рукописного текста и перевод

## 📌 Описание проекта

Этот проект использует **OCR (оптическое распознавание символов)** для извлечения текста с изображения, проверяет его грамматику, а затем переводит на выбранный язык. Графический интерфейс (GUI) реализован с использованием **PySimpleGUI**.

## 🚀 Используемые технологии

- **PySimpleGUI** – создание пользовательского интерфейса
- **EasyOCR** – распознавание текста с изображения
- **OpenCV (cv2)** – предобработка изображений
- **Pillow (PIL)** – работа с изображениями
- **GoogleTranslator (deep\_translator)** – перевод текста
- **language\_tool\_python** – проверка грамматики и орфографии

## 📂 Установка зависимостей

Перед запуском установите необходимые библиотеки:

```sh
pip install PySimpleGUI easyocr opencv-python-headless pillow deep-translator language-tool-python
```

## 🔄 Логика работы кода

### 📌 1. Подготовка изображения (`prepare_image`)

```python
def prepare_image(path):
    res = Image.open(path)
    res.thumbnail((400, 400))
    bio = io.BytesIO()
    res.save(bio, format="PNG")
    return bio.getvalue()
```

- Загружает изображение
- Уменьшает его размер до 400x400 пикселей
- Конвертирует в формат `PNG` для отображения в GUI

### 📌 2. Очистка текста (`cleanup_text`)

```python
def cleanup_text(text):
    return "".join([c if ord(c) < 128 or (1040 <= ord(c) <= 1103) else "" for c in text]).strip()
```

- Убирает не-ASCII символы (кроме русских букв)
- Очищает текст от мусора

### 📌 3. Обработка изображения перед OCR (`preprocess_image`)

```python
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    return blurred
```

- Преобразует изображение в **оттенки серого**
- Применяет бинаризацию (улучшает контраст текста)
- Размывает изображение для удаления шумов

### 📌 4. Распознавание текста (`process`)

```python
def process(path, lang, gpu):
    image = preprocess_image(path)
    reader = Reader([lang], gpu=gpu > 0)
    results = reader.readtext(image)
```

- Загружает изображение
- Использует **EasyOCR** для распознавания текста
- Возвращает найденные слова

### 📌 5. Проверка грамматики (`language_tool_python`)

```python
tool = language_tool_python.LanguageTool('ru')
matches = tool.check(t)
corrected_text = language_tool_python.utils.correct(t, matches)
```

- Проверяет ошибки в тексте
- Исправляет их перед переводом

### 📌 6. Перевод текста (`GoogleTranslator`)

```python
translated_text = GoogleTranslator(source='auto', target=translate_lang).translate(corrected_text)
```

- Определяет исходный язык
- Переводит на выбранный язык

## 🎨 Графический интерфейс (GUI)

```python
layout = [
    [sg.Text('Пожалуйста, выберите изображение:')],
    [sg.InputText(key='-FILEPATH-'), sg.FileBrowse()],
    [sg.Text('Выберите язык текста:'), sg.Combo([...], key='-COMBO-')],
    [sg.Text('Выберите язык перевода:'), sg.Combo([...], key='-TRANSLATE-')],
    [sg.Multiline(key='-TEXT-', size=(50, 5))],
    [sg.Multiline(key='-TRANSLATED-', size=(50, 5))],
    [sg.Image(key='-IMAGE-')],
    [sg.Button('Начать'), sg.Button('Выход'), sg.Button('Помощь')]
]
```

- Позволяет **выбрать изображение**
- Выбор **языка для распознавания** и **перевода**
- Отображает **распознанный и переведённый текст**
- Кнопка **"Помощь"** открывает инструкцию

## ▶️ Запуск программы

```sh
python script.py
```

## 🎯 Итог

Этот проект позволяет **распознавать рукописный текст**, **исправлять ошибки** и **переводить на разные языки** с помощью удобного интерфейса. 🚀

