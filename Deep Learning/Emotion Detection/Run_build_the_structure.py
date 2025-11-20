import os

folders = [
    "notebooks",
    "src/preprocessing",
    "src/model",
    "src/api",
    "src/ui",
    "models/v1",
    "models/v2"
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "src/preprocessing/preprocess.py",
    "src/model/model_architecture.py",
    "src/model/train.py",
    "src/model/evaluate.py",
    "src/api/main.py",
    "src/ui/streamlit_app.py",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, 'w').close()

print("Project structure created successfully!")
