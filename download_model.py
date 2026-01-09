import gdown
import os

def download_model():
    """Download model from Google Drive."""
    
    # Your Google Drive file ID (get from shareable link)
    file_id = "1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq"
    
    # Output path
    output_path = "best_model.npz"
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, output_path, quiet=False)
    print("✅ Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
