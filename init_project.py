import os

def create_private_dir():
    folder_name = "Private"
    path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created folder: {path}")

if __name__ == "__main__":
    create_private_dir()