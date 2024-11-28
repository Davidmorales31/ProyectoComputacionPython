from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import requests

# Cargar el modelo entrenado
model = tf.keras.models.load_model("C:\\Users\\Ana\\PycharmProjects\\PythonProject\\.venv\\final_model.keras")


# Inicializar Flask
app = Flask(__name__)

# Configurar CORS
CORS(app)

# Definir las categorías de maduración
categories = {
    0: 'Inmaduro', 1: 'Inmaduro',
    2: 'Maduración temprana', 3: 'Maduración temprana',
    4: 'Maduración avanzada', 5: 'Maduración avanzada',
    6: 'Maduro', 7: 'Maduro'
}

#URL de la API grupal
url_destino = "https://cdb1-2801-159-0-9227-d0be-215b-d408-908a.ngrok-free.app/insertar"

#Enviar la prediccion a /predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Leer la imagen
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Error reading image'}), 400

    # Preprocesar la imagen
    img = cv2.resize(img, (224, 224))  # Redimensionar a 224x224
    img = np.array(img) / 255.0  # Normalizar la imagen
    img = np.expand_dims(img, axis=0)  # Agregar la dimensión del batch

    # Realizar la predicción
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Asignar la categoría de maduración
    ripeness_level = categories.get(predicted_class, 'Desconocido')

    # Crear el JSON a enviar
    json_data = {
        "fruta_objeto": "Banana",
        "tipo_medicion_id": 2,
        "resultado": ripeness_level,  # El resultado de la predicción
        "estudiante_id": 2
    }

    try:
        response = requests.post(url_destino, json=json_data)
        response.raise_for_status()  # Lanza una excepción si la solicitud falla
        server_response = response.json()
        print(f"JSON enviado correctamente: {json_data}")
        print(f"Respuesta del servidor: {server_response}")
    except requests.RequestException as e:
        server_response = {"error": str(e)}
        print(f"Error al enviar el JSON: {e}")

    # Retornar tanto el resultado local como la respuesta del servidor remoto
    return jsonify({
        "predicted_class": ripeness_level,
        "server_response": server_response
    })


if __name__ == '__main__':
    app.run(debug=True)