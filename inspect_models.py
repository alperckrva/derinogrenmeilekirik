import torch

MODEL_PATHS = {
    "kiriktespit": r"best\kiriktespit.pth",
    "anatomik":    r"best\anatomik.pth",
    "kiriktur":    r"best\kiriktur.pth",
    "kiriksayisi": r"best\kiriksayisi.pt",
}

for name, path in MODEL_PATHS.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}  ({path})")
    print('='*60)
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict):
            keys = list(sd.keys())
            print(f"Toplam katman sayısı: {len(keys)}")
            print(f"\nİlk 5 key:")
            for k in keys[:5]:
                print(f"  {k}: {sd[k].shape}")
            print(f"\nSon 10 key (classifier/fc):")
            for k in keys[-10:]:
                print(f"  {k}: {sd[k].shape}")
        else:
            print(f"Tip: {type(sd)}")
    except Exception as e:
        print(f"HATA: {e}")
