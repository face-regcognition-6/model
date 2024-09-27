from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import numpy as np

class FaceDatabase:
    def __init__(self, face_database):
        self.face_database = face_database
        self.kd_trees = self.build_kd_trees()

    def build_kd_trees(self):
        kd_trees = {}
        for user_id, data in self.face_database.items():
            embeddings = np.array(data['embeddings'])
            if embeddings.shape[0] > 0:
                kd_trees[user_id] = KDTree(embeddings)
        return kd_trees

    def find_nearest(self, embedding):
        min_dist = float('inf')
        identity = 'unknown'
        for user_id, kd_tree in self.kd_trees.items():
            dist, _ = kd_tree.query([embedding], k=1)
            if dist[0][0] < min_dist:
                min_dist = dist[0][0]
                identity = user_id
        return identity, min_dist

def cluster_embeddings(embeddings, min_clusters=2, max_clusters=5):
    """对嵌入向量进行聚类，返回聚类中心"""
    n_clusters = min(max_clusters, max(min_clusters, len(embeddings)))
    if len(embeddings) < min_clusters:
        return embeddings
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_

def process_faces(faces, app, face_db, root):
    CONFIDENCE_THRESHOLD_LOW = 0.0  # 置信度阈值
    
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
            age = face.age
        else:
            gender = face_db.face_database[identity]['gender']
            age = face_db.face_database[identity]['age']

        color = (0, 0, 255)  # Red for unknown or low confidence
        if confidence > 0.5:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.3:
            color = (0, 255, 255)  # Yellow for medium confidence

        face.bbox = bbox
        face.color = color
        face.identity = identity
        face.confidence = confidence
        face.gender = gender
        face.age = age

    return face_confidences
