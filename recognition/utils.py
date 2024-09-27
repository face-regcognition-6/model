import tkinter as tk
from tkinter import simpledialog

def collect_user_data(root, user_exists):
    """弹出窗口收集用户数据"""
    user_data = {}
    user_data['id'] = simpledialog.askstring("Input", "Enter your ID:", parent=root)
    
    if not user_exists:
        user_data['telephoneNo'] = simpledialog.askstring("Input", "Enter your telephone number:", parent=root)
        user_data['gender'] = simpledialog.askstring("Input", "Enter your gender (Male/Female):", parent=root)
        user_data['mail'] = simpledialog.askstring("Input", "Enter your email:", parent=root)
        user_data['age'] = simpledialog.askinteger("Input", "Enter your age:", parent=root)
    else:
        user_data['telephoneNo'] = None
        user_data['gender'] = None
        user_data['mail'] = None
        user_data['age'] = None
    
    return user_data
