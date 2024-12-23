import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('best.pt')

classes = []

with open('./vehicle_dataset/classes.txt', 'r', encoding='utf-8') as file:
    for line in file:
        class_name = line.strip()
        if class_name:
            classes.append(class_name)

print(classes)

def classify_frame_yolo(frame):
    results = model(frame)
    labels = model.names
    found_objects = False

    for i in range(len(results[0].boxes)):
        box = results[0].boxes[i]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        print(labels[class_id])
        if labels[class_id] in classes:
            found_objects = True
            class_name = labels[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, found_objects

@app.route("/", methods=["GET"])
def upload_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Классификация транспортных средств</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                color: #333;
            }
            form {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin: 20px 0;
                width: 300px;
                text-align: center;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #218838;
            }
        </style>
    </head>
    <body>
        <h1>Загрузите изображение для классификации</h1>
        <form action="/process-image" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Загрузить</button>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route("/process-image", methods=["POST"])
def process_image():
    file = request.files['file']
    image_data = file.read()
    np_img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    processed_frame, found_objects = classify_frame_yolo(np_img)

    if not found_objects:
        return render_template_string("""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Результат классификации</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                color: #333;
            }
            h2 {
                color: #555;
                text-align: center;
                margin: 10px 0;
            }
            a {
                text-decoration: none;
                color: #007bff;
                font-weight: bold;
                margin-top: 20px;
                transition: color 0.3s;
            }
            a:hover {
                color: #0056b3;
            }
            .result-container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                text-align: center;
                width: 300px;
            }
        </style>
    </head>
    <body>
        <div class="result-container">
            <h1>Результат обработки изображения</h1>
            <h2>car, threewheel, bus, truck, motorbike, van not found</h2>
            <a href="/">Назад</a>
        </div>
    </body>
    </html>
        """)

    processed_path = "static/processed.jpg"
    os.makedirs("static", exist_ok=True)
    cv2.imwrite(processed_path, processed_frame)

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Результат классификации</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                color: #333;
            }
            h2 {
                color: #555;
                text-align: center;
                margin: 10px 0;
            }
            img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin: 20px 0;
            }
            a {
                text-decoration: none;
                color: #007bff;
                font-weight: bold;
                margin-top: 20px;
                transition: color 0.3s;
            }
            a:hover {
                color: #0056b3;
            }
            .result-container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                text-align: center;
                width: 300px;
            }
        </style>
    </head>
    <body>
        <div class="result-container">
            <h1>Результат обработки изображения</h1>
            <h2>Обработанное изображение</h2>
            <img src="/static/processed.jpg" alt="Processed Image"/>
            <br>
            <a href="/">Назад</a>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)