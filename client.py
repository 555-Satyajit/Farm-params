import streamlit as st
import os
import git
import logging
from typing import List
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_parameters(repo_dir: str, repo_url: str, files: List[str]):
    """Upload model parameters to GitHub repository"""
    try:
        # Initialize repository
        repo = git.Repo.init(repo_dir)
        
        # Add remote
        if 'origin' not in [remote.name for remote in repo.remotes]:
            origin = repo.create_remote('origin', repo_url)
        else:
            origin = repo.remote('origin')
            
        # Add files
        repo.index.add(files)
        
        # Commit changes
        commit_message = "Upload model parameters"
        repo.index.commit(commit_message)
        
        # Push to GitHub
        origin.push('master')
        return True, "Successfully uploaded parameters to GitHub"
        
    except Exception as e:
        return False, f"Error in uploading parameters: {e}"

def main():
    st.set_page_config(page_title="Federated Learning - Client Upload", layout="wide")
    
    st.title("Federated Learning - Client Parameter Upload")
    st.markdown("---")

    # GitHub Repository Configuration
    st.header("GitHub Repository Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter the URL of your GitHub repository"
        )

    # File Upload Section
    st.header("Parameter Files Upload")
    st.markdown("Upload your model parameter files (*.npy files)")
    
    uploaded_files = st.file_uploader(
        "Choose parameter files",
        type=['npy'],
        accept_multiple_files=True,
        help="Select coefficient and intercept files"
    )

    if st.button("Upload to GitHub", type="primary"):
        if not repo_url:
            st.error("Please enter GitHub repository URL")
            return
            
        if not uploaded_files:
            st.error("Please upload parameter files")
            return

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                saved_files.append(uploaded_file.name)

            with st.spinner("Uploading parameters to GitHub..."):
                success, message = upload_parameters(temp_dir, repo_url, saved_files)
                
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)

    # Display instructions
    with st.expander("Instructions"):
        st.markdown("""
        ### How to use this uploader:
        1. Enter your GitHub repository URL
        2. Upload your parameter files (coefficient and intercept .npy files)
        3. Click 'Upload to GitHub' to start the upload process
        
        ### Required Files:
        - Coefficient files (coef_client_X.npy)
        - Intercept files (intercept_client_X.npy)
        
        ### Note:
        Make sure you have the correct GitHub permissions to push to the repository.
        """)

if __name__ == "__main__":
    main()