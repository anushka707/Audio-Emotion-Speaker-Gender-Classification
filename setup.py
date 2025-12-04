import os
import subprocess
import sys

# ---- 1. Create Virtual Environment ----
print("Creating virtual environment 'venv' ...")
subprocess.run([sys.executable, "-m", "venv", "venv"])

print("Virtual environment created.")

# ---- 2. Folder Structure ----
folders = [
    "data",
    "data/ravdess",
    "data/tess",
    "features",
    "models",
    "utils"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ---- 3. Create Empty Files ----
files = [
    "preprocess.py",
    "train.py",
    "predict.py",
    "README.md",
    "requirements.txt",
    "utils/audio_utils.py"
]

for file in files:
    with open(file, "w") as f:
        f.write("")

print("Project structure created successfully.")

# ---- 4. Activation Instructions ----
print("\nNEXT STEP:")
print("Run the following command to activate your virtual environment:\n")
print("source venv/bin/activate")
print("\nThen install dependencies when we generate requirements.txt.")
