import torch

sd = torch.load(r"best\anatomik.pth", map_location='cpu', weights_only=False)
print("Tip:", type(sd))

if isinstance(sd, dict):
    print("Keys:", list(sd.keys()))
    for k, v in sd.items():
        print(f"  '{k}' -> tip: {type(v)}", end="")
        if hasattr(v, 'shape'):
            print(f", shape: {v.shape}")
        elif isinstance(v, dict):
            print(f", alt-key sayısı: {len(v)}")
            # İçindeki son 5 key'i göster
            sub_keys = list(v.keys())
            for sk in sub_keys[-5:]:
                sv = v[sk]
                shape_str = str(sv.shape) if hasattr(sv, 'shape') else str(type(sv))
                print(f"      {sk}: {shape_str}")
        else:
            print(f", değer: {v}")
