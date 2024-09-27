import cv2
from insightface.app import FaceAnalysis
import numpy as np
import mysql.connector
import tkinter as tk
from tkinter import simpledialog
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

# 数据库连接配置
db_config = {
    'user': 'myuser',
    'password': 'mypassword',
    'host': '127.0.0.1',
    'database': 'face_recognition',
    'raise_on_warnings': True
}

root = tk.Tk()
root.withdraw()  # 隐藏主窗口

def cluster_embeddings(embeddings, min_clusters=2, max_clusters=5):
    """对嵌入向量进行聚类，返回聚类中心"""
    n_clusters = min(max_clusters, max(min_clusters, len(embeddings)))
    if len(embeddings) < min_clusters:
        return embeddings
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_

def save_user_data(user_data, embedding, is_new_user):
    """保存用户数据和人脸嵌入向量到数据库"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        if is_new_user:
            # 如果是新用户，插入新用户记录
            add_user = ("INSERT INTO users (id, telephoneNo, gender, mail) VALUES (%s, %s, %s, %s)")
            cursor.execute(add_user, (user_data['id'], user_data['telephoneNo'], user_data['gender'], user_data['mail']))
        
        # 插入人脸嵌入向量
        add_embedding = ("INSERT INTO embeddings (user_id, embedding) VALUES (%s, %s)")
        cursor.execute(add_embedding, (user_data['id'], embedding.tobytes()))
        
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error in save_user_data: {err}")

def load_database():
    """加载数据库中的所有用户数据和嵌入向量"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = ("SELECT u.id, u.telephoneNo, u.gender, u.mail, e.embedding "
                 "FROM users u JOIN embeddings e ON u.id = e.user_id")
        
        cursor.execute(query)
        
        face_database = {}
        for (user_id, telephoneNo, gender, mail, embedding) in cursor:
            embedding = np.frombuffer(embedding, dtype=np.float32)
            if user_id not in face_database:
                face_database[user_id] = {
                    'telephoneNo': telephoneNo,
                    'gender': gender,
                    'mail': mail,
                    'embeddings': []
                }
            face_database[user_id]['embeddings'].append(embedding)
        
        # 聚类每个用户的嵌入向量
        for user_id, data in face_database.items():
            embeddings = np.array(data['embeddings'])
            if len(embeddings) > 1:
                clustered_embeddings = cluster_embeddings(embeddings)
                face_database[user_id]['embeddings'] = clustered_embeddings.tolist()
        
        cursor.close()
        conn.close()
        return face_database
    except mysql.connector.Error as err:
        print(f"Error in load_database: {err}")
        return {}

def check_user_exists(user_id):
    """检查用户ID是否已经存在于数据库中"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = ("SELECT id FROM users WHERE id = %s")
        cursor.execute(query, (user_id,))
        
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return result is not None
    except mysql.connector.Error as err:
        print(f"Error in check_user_exists: {err}")
        return False

def collect_user_data(root, user_exists):
    """弹出窗口收集用户数据"""
    user_data = {}
    user_data['id'] = simpledialog.askstring("Input", "Enter your ID:", parent=root)
    
    if not user_exists:
        user_data['telephoneNo'] = simpledialog.askstring("Input", "Enter your telephone number:", parent=root)
        user_data['gender'] = simpledialog.askstring("Input", "Enter your gender (Male/Female):", parent=root)
        user_data['mail'] = simpledialog.askstring("Input", "Enter your email:", parent=root)
    else:
        user_data['telephoneNo'] = None
        user_data['gender'] = None
        user_data['mail'] = None
    
    return user_data

class FaceDatabase:
    def __init__(self):
        self.face_database = {}
        self.kd_trees = {}

    def load_database(self):
        """加载数据库中的所有用户数据和嵌入向量"""
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            
            query = ("SELECT u.id, u.telephoneNo, u.gender, u.mail, e.embedding "
                     "FROM users u JOIN embeddings e ON u.id = e.user_id")
            
            cursor.execute(query)
            
            for (user_id, telephoneNo, gender, mail, embedding) in cursor:
                embedding = np.frombuffer(embedding, dtype=np.float32)
                if user_id not in self.face_database:
                    self.face_database[user_id] = {
                        'telephoneNo': telephoneNo,
                        'gender': gender,
                        'mail': mail,
                        'embeddings': []
                    }
                self.face_database[user_id]['embeddings'].append(embedding)
            
            # 为每个用户创建KD树
            for user_id, data in self.face_database.items():
                embeddings = np.array(data['embeddings'])
                if embeddings.shape[0] > 0:
                    self.kd_trees[user_id] = KDTree(embeddings)
            
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Error in load_database: {err}")

    def find_nearest(self, embedding):
        min_dist = float('inf')
        identity = 'unknown'
        for user_id, kd_tree in self.kd_trees.items():
            dist, _ = kd_tree.query([embedding], k=1)
            if dist[0][0] < min_dist:
                min_dist = dist[0][0]
                identity = user_id
        return identity, min_dist

face_db = FaceDatabase()
face_db.load_database()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    CONFIDENCE_THRESHOLD_LOW = 0.0  # 置信度阈值
    CONFIDENCE_THRESHOLD_HIGH = 0.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            faces = app.get(frame)

            whiteboard = np.ones_like(frame) * 255
            whiteboard = app.draw_on(whiteboard, faces)

            face_confidences = []
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding
                identity, min_dist = face_db.find_nearest(embedding)
                confidence = 1 - min_dist  # 置信度计算

                face_confidences.append((face, identity, confidence, min_dist, bbox))

                if confidence < CONFIDENCE_THRESHOLD_LOW:
                    identity = 'unknown'
                    gender = 'Male' if face.gender == 1 else 'Female'
                elif CONFIDENCE_THRESHOLD_LOW <= confidence < CONFIDENCE_THRESHOLD_HIGH:
                    gender = 'Male' if face.gender == 1 else 'Female'
                else:
                    gender = face_db.face_database[identity]['gender']

                age = face.age

                color = (0, 0, 255)  # Red for unknown or low confidence
                if confidence > 0.5:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.3:
                    color = (0, 255, 255)  # Yellow for medium confidence

                frame = app.draw_on_dots(frame, faces)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, f'ID: {identity} ({confidence:.2f})', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f'Gender: {gender}, Age: {int(age)}', (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Face Detection and Recognition', frame)
            cv2.imshow('3D Face Model', whiteboard)

            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty('Face Detection and Recognition', cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty('3D Face Model', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('n'):
                if face_confidences:
                    # 找到置信度最低的人脸
                    min_conf_face = min(face_confidences, key=lambda x: x[2])
                    face, identity, confidence, min_dist, bbox = min_conf_face
                    user_id = simpledialog.askstring("Input", "Enter your ID for the face with lowest confidence:", parent=root)
                    if user_id:
                        user_exists = check_user_exists(user_id)
                        user_data = {
                            'id': user_id,
                            'telephoneNo': None,
                            'gender': None,
                            'mail': None
                        }
                        if not user_exists:
                            user_data.update(collect_user_data(root, user_exists))
                        embedding = face.normed_embedding
                        save_user_data(user_data, embedding, not user_exists)
                        face_db.load_database()  # 重新加载数据库
                        print(f"User {user_data['id']} registered successfully with new embedding.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()

if __name__ == "__main__":
    main()
    root.mainloop()
