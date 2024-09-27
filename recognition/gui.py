import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from face_recognition import process_faces, FaceDatabase
from database import load_database
from insightface.app import FaceAnalysis
from app import upload_to_oss
import json
import time

def select_image():
    """选择并读取图片"""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        return cv2.imread(file_path, cv2.IMREAD_COLOR)  # 确保以彩色模式读取
    return None

def resize_image(image, fixed_width=640):
    """将宽度调整到固定大小，高度按比例调整"""
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(fixed_width / aspect_ratio)
    return cv2.resize(image, (fixed_width, new_height), interpolation=cv2.INTER_LINEAR)

def update_image_label(label, image):
    """更新tkinter标签中的图像"""
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=image)
    label.imgtk = imgtk
    label.configure(image=imgtk)

def draw_faces(frame, faces, app):
    whiteboard = np.ones_like(frame) * 255
    whiteboard = app.draw_on(whiteboard, faces)

    for face in faces:
        bbox = face.bbox
        color = face.color
        identity = face.identity
        confidence = face.confidence
        gender = face.gender
        age = face.age

        frame = app.draw_on_dots(frame, [face])
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f'ID: {identity} ({confidence:.2f})', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f'Gender: {gender}, Age: {int(age)}', (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, whiteboard

def main():
    global face_db, left_image_label, right_image_label, face_confidences

    root = tk.Tk()
    root.title("Face Recognition System")

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 12), padding=10)
    style.configure('TLabel', font=('Helvetica', 14))
    style.configure('TFrame', background='#f0f0f0')
    root.configure(bg='#f0f0f0')

    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_db = FaceDatabase(load_database())
    face_confidences = []

    def video_recognition():
        global face_db, face_confidences
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        def process_frame():
            global face_db, face_confidences
            ret, frame = cap.read()
            if not ret:
                cap.release()
                messagebox.showerror("Error", "Can't receive frame (stream end?). Exiting ...")
                return

            faces = app.get(frame)
            face_confidences = process_faces(faces, app, face_db, root)

            frame, whiteboard = draw_faces(frame, faces, app)
            update_image_label(left_image_label, frame)
            update_image_label(right_image_label, whiteboard)

            root.after(10, process_frame)

        process_frame()

    def image_recognition():
        global face_db, face_confidences
        image = select_image()
        if image is None:
            return

        image = resize_image(image)
        faces = app.get(image)

        face_confidences = process_faces(faces, app, face_db, root)

        image, whiteboard = draw_faces(image, faces, app)
        update_image_label(left_image_label, image)
        update_image_label(right_image_label, whiteboard)

    def take_photo():
        global face_confidences
        if face_confidences:
            # 处理所有检测到的人脸
            recognition_results = []
            for face, identity, confidence, min_dist, bbox in face_confidences:
                if confidence < 0:
                    continue  # 跳过置信度小于0的人脸
                user_id = identity
                if user_id != 'unknown':
                    recognition_result = {
                        "user_id": user_id,
                        "identity": identity,
                        "confidence": confidence,
                        "gender": face_db.face_database[identity]['gender'],
                        "age": face_db.face_database[identity]['age']
                    }
                    recognition_results.append(recognition_result)
            
            if recognition_results:
                timestamp = int(time.time())
                file_name = f"recognition_results/{timestamp}.json"
                upload_to_oss(file_name, json.dumps(recognition_results))
                messagebox.showinfo("Success", f"Photo uploaded successfully with results: {json.dumps(recognition_results, indent=2)}")
            else:
                messagebox.showwarning("Warning", "No recognized faces to upload.")
        else:
            messagebox.showwarning("Warning", "No faces detected.")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(main_frame, text="Face Recognition System", background='#f0f0f0', font=('Helvetica', 18, 'bold'))
    title_label.pack(pady=10)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)

    video_button = ttk.Button(button_frame, text="Video Face Recognition", command=video_recognition)
    video_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    image_button = ttk.Button(button_frame, text="Image Face Recognition", command=image_recognition)
    image_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    image_frame = ttk.Frame(main_frame)
    image_frame.pack(fill=tk.BOTH, expand=True)

    left_image_label = ttk.Label(image_frame)
    left_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_image_label = ttk.Label(image_frame)
    right_image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    photo_button = ttk.Button(main_frame, text="Take Photo and Upload", command=take_photo)
    photo_button.pack(fill=tk.X, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
