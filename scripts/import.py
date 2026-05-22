import kagglehub

# Download latest version
path = kagglehub.dataset_download("harrimansaragih/dummy-advertising-and-sales-data")

print("Path to dataset files:", path)