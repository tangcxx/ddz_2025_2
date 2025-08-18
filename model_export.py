#%%
import torch
import bot_torch_ln

#%%
def export(id):
        cp = torch.load(f"model_tree2/cp{id}.pt")
        dummy_input = torch.randn(1, 15, 3, 15)
        for pos in range(3):
            model = bot_torch_ln.Model()
            model.load_state_dict(cp["models_state_dict"][pos])
            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                f"onnx/model_{pos}_{id}.onnx",
                input_names=['input'],
                output_names=['output'],
                opset_version=13,
                dynamic_axes={
                    'input': {0: 'batch_size'},   # Define the first dimension as dynamic
                    'output': {0: 'batch_size'}   # Similarly for outputs
                }
)
    
#%%
export(85570)


# %%
