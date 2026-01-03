# graph.py
import argparse
import torch
from torchviz import make_dot
from model import LSTMRegressor


def load_model(model_path):
    """
    Model ağırlığını yükler ve aynı mimariyi oluşturur.
    Mimariyi sen istediğin gibi değiştireceksen bu fonksiyonda oynarsın.
    """
    model = LSTMRegressor(
        input_dim=5,     # sabit: X feature sayısı
        hidden=64,       # burada istersen değiştirebilirsin
        layers=2,
        dropout=0.1,
        out_dim=3
    )

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def visualize(model, timesteps=31, features=5):
    """
    Dummy input ile computational graph oluşturup PNG kaydeder.
    """
    dummy = torch.randn(1, timesteps, features)
    pred = model(dummy)

    dot = make_dot(pred, params=dict(model.named_parameters()))
    dot.format = "png"
    output_path = "model_graph"
    dot.render(output_path, cleanup=True)

    print(f"✔ Grafik oluşturuldu: {output_path}.png")


def main():
    parser = argparse.ArgumentParser(description="PyTorch model graph generator")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="outputs\win31_hid64_layers2_bs32_lr0.0005.pt"
    )
    args = parser.parse_args()

    print(f"[INFO] Model yükleniyor: {args.model}")
    model = load_model(args.model)

    print("[INFO] Grafik oluşturuluyor...")
    visualize(model)


if __name__ == "__main__":
    main()
