import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import base64
import io

app = Flask(__name__)

def detect_faces(img_stream):
    try:
        # Convert the image stream to a numpy array
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Load the pre-trained face detection model from the root directory
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around detected faces
        img_contours = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Loop over the detected faces and blur each one
        img_blur = img.copy()
        for (x, y, w, h) in faces:
            face_region = img_blur[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            img_blur[y:y+h, x:x+w] = face_region
        
        # Encode processed images to base64 strings
        _, encoded_img_contours = cv2.imencode('.jpg', img_contours)
        base64_img_contours = base64.b64encode(encoded_img_contours).decode('utf-8')

        _, encoded_img_blur = cv2.imencode('.jpg', img_blur)
        base64_img_blur = base64.b64encode(encoded_img_blur).decode('utf-8')

        return base64_img_contours, base64_img_blur

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        # Process the uploaded image to detect faces and create two versions
        base64_img_contours, base64_img_blur = detect_faces(file)
        
        if base64_img_contours and base64_img_blur:
            return jsonify({
                'success': True,
                'img_contours': base64_img_contours,
                'img_blur': base64_img_blur
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to process image'})
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'message': 'Error processing image'})

@app.route('/download/<version>', methods=['GET'])
def download_image(version):
    try:
        file = request.args.get('file')
        img_data = base64.b64decode(file)
        img_io = io.BytesIO(img_data)
        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name=f'{version}.jpg')
    except Exception as e:
        print(f"Error downloading image: {e}")
        return jsonify({'success': False, 'message': 'Error downloading image'})

if __name__ == '__main__':
    app.run(debug=True)
