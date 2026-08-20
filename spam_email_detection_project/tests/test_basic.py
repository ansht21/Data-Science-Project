"""
Basic test for spam email detection project structure.
"""
import os
import sys

def test_directories_exist():
    """Test that expected directories exist."""
    base_path = os.path.dirname(__file__)
    
    # Test src directory
    src_path = os.path.join(base_path, '..', 'src')
    assert os.path.exists(src_path), f"SRC directory not found: {src_path}"
    
    # Test data directory
    data_path = os.path.join(base_path, '..', 'data')
    assert os.path.exists(data_path), f"DATA directory not found: {data_path}"
    
    # Test models directory
    models_path = os.path.join(base_path, '..', 'models')
    assert os.path.exists(models_path), f"MODELS directory not found: {models_path}"
    
    # Test notebooks directory
    notebooks_path = os.path.join(base_path, '..', 'notebooks')
    assert os.path.exists(notebooks_path), f"NOTEBOOKS directory not found: {notebooks_path}"
    
    print("All directory structure tests passed!")

def test_required_files_exist():
    """Test that required files exist."""
    base_path = os.path.dirname(__file__)
    
    # Test requirements.txt
    req_path = os.path.join(base_path, '..', 'requirements.txt')
    assert os.path.exists(req_path), f"Requirements file not found: {req_path}"
    
    # Test README.md
    readme_path = os.path.join(base_path, '..', 'README.md')
    assert os.path.exists(readme_path), f"README file not found: {readme_path}"
    
    # Test that notebook exists
    notebook_path = os.path.join(base_path, '..', 'notebooks', 'spam_email_detection.ipynb')
    assert os.path.exists(notebook_path), f"Notebook not found: {notebook_path}"
    
    print("All required files tests passed!")

if __name__ == "__main__":
    test_directories_exist()
    test_required_files_exist()
    print("All basic tests passed!")
