import numpy as np
import mysql.connector

# 数据库连接配置
db_config = {
    'user': 'myuser',
    'password': 'mypassword',
    'host': '127.0.0.1',
    'database': 'face_recognition',
    'raise_on_warnings': True
}

def connect_to_database():
    return mysql.connector.connect(**db_config)

def save_user_data(user_data, embedding, is_new_user):
    """保存用户数据和人脸嵌入向量到数据库"""
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        
        if is_new_user:
            # 如果是新用户，插入新用户记录
            add_user = ("INSERT INTO users (id, telephoneNo, gender, mail, age) VALUES (%s, %s, %s, %s, %s)")
            cursor.execute(add_user, (user_data['id'], user_data['telephoneNo'], user_data['gender'], user_data['mail'], user_data['age']))
        
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
        conn = connect_to_database()
        cursor = conn.cursor()
        
        query = ("SELECT u.id, u.telephoneNo, u.gender, u.mail, u.age, e.embedding "
                 "FROM users u JOIN embeddings e ON u.id = e.user_id")
        
        cursor.execute(query)
        
        face_database = {}
        for (user_id, telephoneNo, gender, mail, age, embedding) in cursor:
            embedding = np.frombuffer(embedding, dtype=np.float32)
            if user_id not in face_database:
                face_database[user_id] = {
                    'telephoneNo': telephoneNo,
                    'gender': gender,
                    'mail': mail,
                    'age': age,
                    'embeddings': []
                }
            face_database[user_id]['embeddings'].append(embedding)
        
        cursor.close()
        conn.close()
        return face_database
    except mysql.connector.Error as err:
        print(f"Error in load_database: {err}")
        return {}

def check_user_exists(user_id):
    """检查用户ID是否已经存在于数据库中"""
    try:
        conn = connect_to_database()
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
