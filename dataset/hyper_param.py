hyperParamsDict = {
    "BATCH_SIZE" : 64,
    "EPOCHS" : 50,
    "IMAGE_SIZE": (224, 224),
    "LR" : 1e-3,
    "FOLDS" : 5,
    "WEIGHT_DECAY": 0.01,
    "SEED": 13,
    "NUM_WORKERS": 4,
    "LOSS_FUNC" : "CrossEntropy",
    "OPTIMIZER_NAME" : "AdamW",
}

# Transfer learning: mesmos valores do baseline, exceto épocas e LR
transferHyperParamsDict = {
    **hyperParamsDict,
    "EPOCHS": 15,  # modelos pré-treinados convergem bem antes de 50 épocas
    # LR menor quando o backbone é treinado, para não destruir os pesos do ImageNet
    "LR_BY_STRATEGY": {
        "frozen": 1e-3,
        "partial": 1e-4,
        "full": 1e-4,
        "scratch": 1e-3,
    },
}
