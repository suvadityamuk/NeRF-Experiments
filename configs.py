NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16

exp1_config = {
    "NUM_GPUS": 1,
    "POS_ENCODE_DIMS": 16,
    "NUM_SAMPLES": NUM_SAMPLES,
    "BATCH_SIZE": 4,
    "H": 100,
    "W": 100,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.00,
    "EPOCHS": 50,
    "AMSGRAD": False,
    "FUSED_OPTIM": False,
    "EMB_DIM": 256,
}

exp2_config = {
    "NUM_GPUS": 1,
    "POS_ENCODE_DIMS": 16,
    "NUM_SAMPLES": NUM_SAMPLES,
    "BATCH_SIZE": 4,
    "H": 100,
    "W": 100,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.00,
    "EPOCHS": 50,
    "AMSGRAD": True,
    "FUSED_OPTIM": True,
    "EMB_DIM": 256,
}

exp3_config = {
    "NUM_GPUS": 1,
    "POS_ENCODE_DIMS": 16,
    "NUM_SAMPLES": NUM_SAMPLES,
    "BATCH_SIZE": 8,
    "H": 100,
    "W": 100,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.05,
    "EPOCHS": 50,
    "AMSGRAD": True,
    "FUSED_OPTIM": True,
    "EMB_DIM": 256,
}

exp4_config = {
    "NUM_GPUS": 1,
    "POS_ENCODE_DIMS": 16,
    "NUM_SAMPLES": NUM_SAMPLES,
    "BATCH_SIZE": 4,
    "H": 100,
    "W": 100,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 0.05,
    "EPOCHS": 50,
    "AMSGRAD": True,
    "FUSED_OPTIM": True,
    "EMB_DIM": 256,
}

exp5_config = {
    "NUM_GPUS": 2,
    "NUM_SAMPLES": 32,
    "BATCH_SIZE": 3,
    "H": 100,
    "W": 100,
    "POS_ENCODE_DIMS": 16,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.00,
    "EPOCHS": 50,
    "AMSGRAD": False,
    "FUSED_OPTIM": False,
    "EMB_DIM": 256,
}

exp6_config = {
    "NUM_GPUS": 2,
    "NUM_SAMPLES": 32,
    "BATCH_SIZE": 2,
    "H": 100,
    "W": 100,
    "POS_ENCODE_DIMS": 16,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.00,
    "EPOCHS": 50,
    "AMSGRAD": False,
    "FUSED_OPTIM": False,
    "EMB_DIM": 256,
}
