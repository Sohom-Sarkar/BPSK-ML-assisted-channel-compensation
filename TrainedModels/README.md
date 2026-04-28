# Trained Model Checkpoints

Pre-trained weights for all six ML models plus two baselines.
PyTorch checkpoints (.pt):

    import torch
    ckpt = torch.load('mlp_iq.pt', weights_only=False, map_location='cpu')
    # keys: state_dict, best_cfg, scaler_mean, scaler_scale

sklearn checkpoints (.pkl):

    import pickle
    with open('linear_iq.pkl', 'rb') as f:
        obj = pickle.load(f)   # dict with key 'model' -> sklearn Pipeline

Models
    mlp_iq.pt           MLP on raw I/Q window              input dim = 22
    mlp_phase.pt        MLP on differential-phase window    input dim = 11
    mlp_combined.pt     MLP on combined features            input dim = 43
    resnet_combined.pt  ResNet on combined features         input dim = 43
    cvnn.pt             Complex-valued NN                   input dim = 11 cmplx
    lstm.pt             LSTM sequence equaliser             input dim = 2 per step
    linear_iq.pkl       Ridge regression                    input dim = 22
    linear.pkl          Alternative ridge checkpoint        input dim = 22

For ready-to-use inference examples see Python/demo_live.py.
Model class definitions are in Python/common.py.
