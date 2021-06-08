from architectures.unet.encoder import UNetEncoder
import torch


def test_UNetEncoder():

    t = torch.randn(size=(2, 3, 256, 256))
    channels_enc = [64, 64, 128, 256, 512]
    unet_encoder = UNetEncoder(channels_enc)

    features = unet_encoder(t)

    for lvl in range(len(channels_enc)):
        print(features[lvl].shape)
        assert features[lvl].size(1) == channels_enc[lvl]
        assert features[lvl].size(2) == 256 / (2 ** lvl)
        assert features[lvl].size(3) == 256 / (2 ** lvl)
