import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from PIL import Image
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['AugTheft', 'Nil']

# Custom Dataset for test images (no labels required)
class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Create Vision Transformer model
def create_vit_model(num_classes=NUM_CLASSES):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)

# Test and save classified images
def test_and_save(model, test_loader, output_dir):
    model.eval()
    os.makedirs(os.path.join(output_dir, 'AugTheft'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Nil'), exist_ok=True)

    aug_theft_count = 0
    nil_count = 0

    with torch.no_grad():
        progress = st.progress(0)
        total_batches = len(test_loader)
        for batch_idx, (images, img_paths) in enumerate(test_loader):
            images = images.to(DEVICE)
            with autocast(DEVICE.type):
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

            for img_path, pred in zip(img_paths, predicted):
                class_name = CLASS_NAMES[pred.item()]
                dest_path = os.path.join(output_dir, class_name, os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
                if class_name == 'AugTheft':
                    aug_theft_count += 1
                else:
                    nil_count += 1

            progress.progress((batch_idx + 1) / total_batches)

    return aug_theft_count, nil_count

# Function to create a zip file of the output directory
def create_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

# Streamlit UI
def main():
    st.set_page_config(page_title="Meter Classification App", layout="wide", initial_sidebar_state="expanded")

    
    st.markdown("""
        <style>
        .stApp {
            background-color: #F8F9FA;
            color: #212529;
        }
        .main .block-container {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem;
        }
        /* Sidebar styling - dark background with light text */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #2C3E50 !important;
        }
        .css-1d391kg .stMarkdown, 
        [data-testid="stSidebar"] .stMarkdown,
        .css-1d391kg .element-container,
        [data-testid="stSidebar"] .element-container,
        .css-1d391kg h1,
        [data-testid="stSidebar"] h1,
        .css-1d391kg p,
        [data-testid="stSidebar"] p,
        .css-1d391kg .stCaption,
        [data-testid="stSidebar"] .stCaption {
            color: #FFFFFF !important;
        }
        .stButton > button {
            background-color: #007BFF;
            color: #FFFFFF;
            border: none;
            padding: 12px 28px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            font-weight: bold;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 6px;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #0056B3;
        }
        .stTextInput > div > div > input {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CED4DA;
            border-radius: 4px;
        }
        /* Main content styling - light background with dark text */
        .stMarkdown {
            color: #212529 !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #212529 !important;
        }
        .stHeader {
            color: #212529 !important;
        }
        .stSubheader {
            color: #212529 !important;
        }
        .element-container {
            color: #212529 !important;
        }
        [data-testid="stFileUploader"] {
            color: #212529 !important;
        }
        [data-testid="stFileUploader"] label {
            color: #212529 !important;
        }
        [data-testid="stFileUploader"] div {
            color: #212529 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Meter Classification")
        st.markdown("Upload your dataset of meter images for theft detection classification.")
        st.markdown("---")
        model_path = st.text_input("Model Path", value="vit_model.pth", help="Path to the trained ViT model file.")
        st.markdown("---")
        st.caption(f"Using device: {DEVICE.type.upper()}")
        if DEVICE.type == "cuda":
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Upload Dataset")
        uploaded_files = st.file_uploader("Choose image files (JPG, PNG)", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded successfully.")

            if st.button("Classify Images", key="classify_btn"):
                if not os.path.exists(model_path):
                    st.error("Model file not found. Please provide a valid path.")
                    return

                with tempfile.TemporaryDirectory() as temp_test_dir:
                    with tempfile.TemporaryDirectory() as temp_output_dir:
                        # Save uploaded files to temp directory
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_test_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                        # Image transformations
                        transform = transforms.Compose([
                            transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

                        # Create and load model
                        model = create_vit_model()
                        try:
                            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
                            return

                        # Test dataset and loader
                        test_dataset = TestImageDataset(temp_test_dir, transform=transform)
                        if len(test_dataset) == 0:
                            st.error("No valid images found in the uploaded files.")
                            return

                        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                        # Classify and save images
                        st.info("Classifying images...")
                        aug_theft_count, nil_count = test_and_save(model, test_loader, temp_output_dir)

                        # Create zip file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_filename = f"classified_output_{timestamp}.zip"
                        
                        # Create zip in memory
                        zip_buffer = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
                        for root, _, files in os.walk(temp_output_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_output_dir)
                                zip_buffer.write(file_path, arcname)
                        zip_buffer.close()

                        # Display results
                        st.success("Classification complete!")
                        st.markdown(f"**AugTheft:** {aug_theft_count} images")
                        st.markdown(f"**Nil:** {nil_count} images")
                        st.markdown(f"**Total:** {aug_theft_count + nil_count} images")

                        # Download button
                        with open(zip_filename, "rb") as f:
                            st.download_button(
                                label="Download Classified Output (ZIP)",
                                data=f.read(),
                                file_name=zip_filename,
                                mime="application/zip"
                            )
                        
                        # Clean up zip file
                        os.remove(zip_filename)

    with col2:
        st.header("About")
        st.markdown("""
        This app uses a Vision Transformer (ViT) model to classify meter images into:
        - **AugTheft**: Indicating potential theft augmentation.
        - **Nil**: No issues detected.
        
        Upload multiple images, classify them, and download the results organized into folders.
        """)
        st.markdown("---")
        st.subheader("Instructions")
        st.markdown("""
        1. Provide the path to your trained model (vit_model.pth).
        2. Upload your image files.
        3. Click 'Classify Images'.
        4. Download the ZIP file containing classified folders.
        """)

if __name__ == '__main__':
    main()