# app.py
import oss2
import json

# 阿里云OSS配置
access_key_id = 'LTAI5t83aauvbnbfKXtdf1x9'
access_key_secret = 'jpRq1CA1Pa64eU9YOQkxryPsZ93tnP'
bucket_name = 'sky-facerecognition'
endpoint = 'oss-cn-beijing.aliyuncs.com'

# 创建OSS Bucket对象
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def upload_to_oss(file_name, data):
    """上传数据到阿里云OSS"""
    try:
        result = bucket.put_object(file_name, data)
        if result.status == 200:
            print(f"Successfully uploaded to OSS: {file_name}")
        else:
            print(f"Failed to upload to OSS: {file_name}, status: {result.status}")
    except oss2.exceptions.ServerError as e:
        print(f"ServerError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
