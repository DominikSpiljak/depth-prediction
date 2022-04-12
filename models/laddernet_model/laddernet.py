import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DENSENETS = {
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
}

DIMENSIONALITIES = {
    "densenet121": [
        {"db": 1024},
        {"db": 1024, "context_in": 256, "context_out": 256},
        {"db": 640, "context_in": 256, "context_out": 256},
        {"db": 512, "context_in": 256, "context_out": 128},
        {"db": 256, "context_in": 128, "context_out": 128},
    ],
    "densenet161": [
        {"db": 2208},
        {"db": 2112, "context_in": 552, "context_out": 256},
        {"db": 1248, "context_in": 256, "context_out": 256},
        {"db": 768, "context_in": 256, "context_out": 128},
        {"db": 384, "context_in": 128, "context_out": 128},
    ],
    "densenet169": [
        {"db": 1664},
        {"db": 1280, "context_in": 416, "context_out": 256},
        {"db": 768, "context_in": 256, "context_out": 256},
        {"db": 512, "context_in": 256, "context_out": 128},
        {"db": 256, "context_in": 128, "context_out": 128},
    ],
    "densenet201": [
        {"db": 1920},
        {"db": 1792, "context_in": 480, "context_out": 256},
        {"db": 1024, "context_in": 256, "context_out": 256},
        {"db": 512, "context_in": 256, "context_out": 128},
        {"db": 256, "context_in": 128, "context_out": 128},
    ],
}


class PoolConvUpscale(nn.Module):
    def __init__(self, input_features, grid_rows):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=input_features,
            out_channels=input_features // 4,
            kernel_size=1,
            padding="same",
        )
        self.grid_rows = grid_rows

    def forward(self, x):
        original_res = x.shape[2:]
        x = F.adaptive_avg_pool2d(
            x, (self.grid_rows, self.grid_rows * x.shape[3] // x.shape[2])
        )
        x = self.conv(x)
        return F.interpolate(x, size=original_res, mode="bilinear")


class SpatialPyramidPooling(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=input_features,
            out_channels=input_features // 2,
            kernel_size=1,
            padding="same",
        )
        self.poolconvup1 = PoolConvUpscale(input_features // 2, grid_rows=1)
        self.poolconvup2 = PoolConvUpscale(input_features // 2, grid_rows=2)
        self.poolconvup4 = PoolConvUpscale(input_features // 2, grid_rows=4)
        self.poolconvup8 = PoolConvUpscale(input_features // 2, grid_rows=8)
        self.blend_conv = nn.Conv2d(
            in_channels=input_features,
            out_channels=input_features // 4,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        x = self.projection(x)
        x1 = self.poolconvup1(x)
        x2 = self.poolconvup2(x)
        x4 = self.poolconvup4(x)
        x8 = self.poolconvup8(x)
        x = torch.cat((x, x1, x2, x4, x8), dim=1)
        return self.blend_conv(x)


class TransitionUpBlock(nn.Module):
    def __init__(self, dbdim, contextdim_in, contextdim_out):
        super().__init__()
        self.db_conv = nn.Conv2d(
            in_channels=dbdim, out_channels=contextdim_in, kernel_size=1, padding="same"
        )
        self.context_convs = nn.Sequential(
            nn.Conv2d(
                in_channels=contextdim_in,
                out_channels=contextdim_in // 2,
                kernel_size=1,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=contextdim_in // 2,
                out_channels=contextdim_out,
                kernel_size=3,
                padding="same",
            ),
        )

    def forward(self, db_out, context):
        context = F.interpolate(context, size=db_out.shape[2:], mode="bilinear")
        db_out = self.db_conv(db_out)
        return self.context_convs(context + db_out)


class PartialDenseBlock(nn.Module):
    def __init__(self, dense_layers):
        super().__init__()
        self.dense_layers = nn.ModuleList(dense_layers)

    def forward(self, x):
        features = [x]
        for layer in self.dense_layers:
            out = layer(features)
            features.append(out)
        return torch.cat(features, dim=1)


class DenseNet(nn.Module):
    def __init__(self, densenet="densenet121", pretrained=True):
        super().__init__()
        densenet_layers = list(DENSENETS[densenet](pretrained=pretrained).children())[
            :-1
        ][0]

        self.stem = nn.Sequential(*densenet_layers[:4])
        self.db1 = densenet_layers[4]
        self.td1 = densenet_layers[5]
        self.db2 = densenet_layers[6]
        self.td2 = densenet_layers[7]
        db3_layers = list(densenet_layers[8].children())
        self.db3a = PartialDenseBlock(db3_layers[: len(db3_layers) // 2])
        self.d3 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            ceil_mode=False,
            count_include_pad=False,
        )
        self.db3b = PartialDenseBlock(db3_layers[len(db3_layers) // 2 :])
        self.td3 = densenet_layers[9]
        self.db4 = densenet_layers[10]

    def forward(self, x):
        x = self.stem(x)
        db1_out = self.db1(x)
        x = self.td1(db1_out)
        db2_out = self.db2(x)
        x = self.td2(db2_out)
        db3a_out = self.db3a(x)
        x = self.d3(db3a_out)
        db3b_out = self.db3b(x)
        x = self.td3(db3b_out)
        return db1_out, db2_out, db3a_out, db3b_out, self.db4(x)


class LadderNet(nn.Module):
    def __init__(self, densenet="densenet121", pretrained=True):
        super().__init__()

        dimensionality_config = DIMENSIONALITIES[densenet]

        self.densenet = DenseNet(densenet, pretrained)
        self.spp = SpatialPyramidPooling(input_features=dimensionality_config[0]["db"])
        self.tu3b = TransitionUpBlock(
            dbdim=dimensionality_config[1]["db"],
            contextdim_in=dimensionality_config[1]["context_in"],
            contextdim_out=dimensionality_config[1]["context_out"],
        )
        self.tu3a = TransitionUpBlock(
            dbdim=dimensionality_config[2]["db"],
            contextdim_in=dimensionality_config[2]["context_in"],
            contextdim_out=dimensionality_config[2]["context_out"],
        )
        self.tu2 = TransitionUpBlock(
            dbdim=dimensionality_config[3]["db"],
            contextdim_in=dimensionality_config[3]["context_in"],
            contextdim_out=dimensionality_config[3]["context_out"],
        )
        self.tu1 = TransitionUpBlock(
            dbdim=dimensionality_config[4]["db"],
            contextdim_in=dimensionality_config[4]["context_in"],
            contextdim_out=dimensionality_config[4]["context_out"],
        )

        self.out_map_spp = nn.Conv2d(
            in_channels=dimensionality_config[1]["context_in"],
            out_channels=1,
            kernel_size=1,
            padding="same",
        )
        self.out_map_tu3b = nn.Conv2d(
            in_channels=dimensionality_config[1]["context_out"],
            out_channels=1,
            kernel_size=1,
            padding="same",
        )
        self.out_map_tu3a = nn.Conv2d(
            in_channels=dimensionality_config[2]["context_out"],
            out_channels=1,
            kernel_size=1,
            padding="same",
        )
        self.out_map_tu2 = nn.Conv2d(
            in_channels=dimensionality_config[3]["context_out"],
            out_channels=1,
            kernel_size=1,
            padding="same",
        )
        self.out_map_tu1 = nn.Conv2d(
            in_channels=dimensionality_config[4]["context_out"],
            out_channels=1,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        db1_out, db2_out, db3a_out, db3b_out, db4_out = self.densenet(x)
        spp_out = self.spp(db4_out)
        tu3b_out = self.tu3b(db3b_out, spp_out)
        tu3a_out = self.tu3a(db3a_out, tu3b_out)
        tu2_out = self.tu2(db2_out, tu3a_out)
        tu1_out = self.tu1(db1_out, tu2_out)

        return (
            F.interpolate(self.out_map_tu1(tu1_out), scale_factor=4),
            self.out_map_tu2(tu2_out),
            self.out_map_tu3a(tu3a_out),
            self.out_map_tu3b(tu3b_out),
            self.out_map_spp(spp_out),
        )


if __name__ == "__main__":
    input = torch.randn(size=(1, 3, 240, 320))
    laddernet = LadderNet(densenet="densenet201")

    out, tu2, tu3a, tu3b, spp = laddernet(input)

    print(out.shape, tu2.shape, tu3a.shape, tu3b.shape, spp.shape)
