import kagglehub

# Download latest version
path = kagglehub.dataset_download("robikscube/eye-state-classification-eeg-dataset")

print("Path to dataset files:", path)
