from models.FlexibleCNN import torch_load_compat
p = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Comparacion entre agentes\Agentes\epoch_324.pt"
print("Intentando cargar:", p)
try:
    obj = torch_load_compat(p)
    print("Carga OK. Tipo cargado:", type(obj))
    try:
        keys = list(obj.keys())
        print("Es dict. Primeras 10 claves:", keys[:10])
    except Exception:
        print("Objeto cargado no es dict o no tiene .keys()")
except Exception as e:
    import traceback
    print("Error al cargar checkpoint:")
    traceback.print_exc()

