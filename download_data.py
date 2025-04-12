import kagglehub

# Download latest version
path = kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")

print("Path to dataset files:", path)