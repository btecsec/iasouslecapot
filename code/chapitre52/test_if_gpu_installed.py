import torch
print(torch.cuda.is_available())      # doit renvoyer True
print(torch.cuda.get_device_name(0))  # doit afficher "NVIDIA T1200 Laptop GPU"