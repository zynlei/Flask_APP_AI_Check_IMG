from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load model và nhãn từ file
model = load_model("keras_model.h5")
# Nhãn của mô hình
labels = ['san-pham-phong-trang', 'san-pham-mau', 'thong-so', 'cau-tao-nguyen-ly']

# Hàm xử lý ảnh
def preprocess_image(image):
    # Kiểm tra nếu ảnh không phải là RGB, chuyển đổi sang RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))  # Chỉnh lại kích thước ảnh nếu cần
    image = np.array(image) / 255.0  # Chuẩn hóa ảnh
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension
    return image

# Hàm tải ảnh từ URL
def load_image_from_url(image_url):
    try:
        # Gửi yêu cầu GET tới URL để lấy hình ảnh
        response = requests.get(image_url)
        response.raise_for_status()  # Kiểm tra mã trạng thái của yêu cầu
        image = Image.open(BytesIO(response.content))  # Mở hình ảnh từ dữ liệu nhận được
        return image
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error loading image from URL: {e}")
    except Exception as e:
        raise Exception(f"Error processing the image: {e}")

# Endpoint API
@app.route("/predict", methods=["POST"])
def predict():
    # Lấy URL hình ảnh từ dữ liệu JSON trong body của yêu cầu
    image_url = request.json.get('image_url', None)
    
    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400
    
    try:
        # Tải hình ảnh từ URL
        image = load_image_from_url(image_url)
        
        # Tiền xử lý và dự đoán
        image = preprocess_image(image)
        predictions = model.predict(image)
        
        # Xử lý kết quả dự đoán
        class_idx = np.argmax(predictions, axis=1)[0]
        class_label = labels[class_idx]  # Sử dụng nhãn tương ứng
        confidence = predictions[0][class_idx]
        
        # Trả về kết quả dự đoán
        return jsonify({"label": class_label, "confidence": float(confidence)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lưu ý: Không cần gọi app.run() vì Vercel sẽ tự động xử lý việc này.
