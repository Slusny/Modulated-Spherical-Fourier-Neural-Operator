

def test_autoregressive_forecast(checkpoint_list,hyperparams):
    for checkpoint in checkpoint_list:
        print(f"Testing checkpoint {checkpoint}")
        model = S2SModel(hyperparams)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        model = model.to(device)
        model.autoregressive_forecast()
        print("Test passed")
