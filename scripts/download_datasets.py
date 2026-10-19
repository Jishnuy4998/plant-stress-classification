import kagglehub

# Training dataset: PlantVillage
plantvillage_path = kagglehub.dataset_download(
    "emmarex/plantdisease",
    output_dir="./data/downloads/plantvillage"
)
print("PlantVillage downloaded to:", plantvillage_path)

# External validation dataset: PlantDoc
plantdoc_path = kagglehub.dataset_download(
    "abdulhasibuddin/plant-doc-dataset",
    output_dir="./data/downloads/plantdoc"
)
print("PlantDoc downloaded to:", plantdoc_path)