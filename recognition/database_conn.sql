CREATE DATABASE face_recognition;

USE face_recognition;

CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    telephoneNo VARCHAR(15),
    gender VARCHAR(10),
    mail VARCHAR(120)
);

CREATE TABLE embeddings (
    user_id VARCHAR(36),
    embedding BLOB,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

ALTER TABLE users ADD COLUMN age INT;
