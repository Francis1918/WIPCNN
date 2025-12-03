import os, traceback
p = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Comparacion entre agentes\Agentes\epoch_324.pt"
print("CHECKPOINT PATH:", p)
print("Exists:", os.path.exists(p))
if os.path.exists(p):
    print("Size:", os.path.getsize(p))

try:
    import torch
    print("Torch version:", torch.__version__)
except Exception as e:
    print("Could not import torch:", e)

try:
    from models.FlexibleCNN import torch_load_compat
    print("Imported torch_load_compat ok")
except Exception as e:
    print("Error importing torch_load_compat:")
    traceback.print_exc()

print('\n-- Calling torch_load_compat --')
try:
    obj = torch_load_compat(p)
    print('Loaded object type:', type(obj))
    try:
        print('Sample keys:', list(obj.keys())[:10])
    except Exception:
        print('Loaded object not a dict or no keys()')
except Exception as e:
    print('Exception while loading:')
    traceback.print_exc()

print('\n-- Done --')

