import math
import types

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dpt_model.utils import match_weights


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output

    return hook


def resize_for_pos_embed(pos_embed, num_toks_h, num_toks_w):
    posemb_tok, posemb_grid = (
        pos_embed[:, :1],
        pos_embed[0, 1:],
    )

    num_toks_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, num_toks_old, num_toks_old, -1).permute(
        0, 3, 1, 2
    )
    posemb_grid = F.interpolate(
        posemb_grid, size=(num_toks_h, num_toks_w), mode="bilinear"
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, num_toks_h * num_toks_w, -1
    )

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def hybrid_forward(self, x):
    b, c, w, h = x.shape

    pos_embed = resize_for_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    x = self.patch_embed(x)

    cls_token = self.cls_token.expand(b, -1, -1)

    x = torch.cat((cls_token, x), dim=1)

    x = self.pos_drop(x + pos_embed)
    x = self.blocks(x)
    return self.norm(x)


def backbone_forward(self, x):
    b, c, h, w = x.shape
    self.hybrid_forward(x)

    return (
        self.activations["1"],
        self.activations["2"],
        self.act_postprocess3(self.activations["3"]),
        self.act_postprocess4(self.activations["4"]),
    )


class ProjectReadout(nn.Module):
    def __init__(self, in_features):
        super(ProjectReadout, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.GELU(),
        )

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, 1:])
        features = torch.cat((x[:, 1:], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


def create_vit_hybrid_backbone(
    pretrained,
    hooks=None,
    size=None,
):
    backbone = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    backbone.patch_size = [16, 16]
    backbone.activations = {}

    hooks = hooks if hooks else [0, 1, 8, 11]
    size = size if size else [320, 240]

    backbone.patch_embed.backbone.stages[hooks[0]].register_forward_hook(
        get_activation("1", backbone.activations)
    )
    backbone.patch_embed.backbone.stages[hooks[1]].register_forward_hook(
        get_activation("2", backbone.activations)
    )
    backbone.blocks[hooks[2]].register_forward_hook(
        get_activation("3", backbone.activations)
    )
    backbone.blocks[hooks[3]].register_forward_hook(
        get_activation("4", backbone.activations)
    )

    backbone.act_postprocess3 = nn.Sequential(
        ProjectReadout(768),
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    backbone.act_postprocess4 = nn.Sequential(
        ProjectReadout(768),
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    backbone.hybrid_forward = types.MethodType(hybrid_forward, backbone)
    backbone.forward = types.MethodType(backbone_forward, backbone)
    return backbone


def create_scratch():
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        256,
        256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=1,
    )
    scratch.layer2_rn = nn.Conv2d(
        512,
        256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=1,
    )
    scratch.layer3_rn = nn.Conv2d(
        768,
        256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=1,
    )
    scratch.layer4_rn = nn.Conv2d(
        768,
        256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=1,
    )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=1,
        )

        self.conv2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=1,
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.out_conv = nn.Conv2d(
            256,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit()
        self.resConfUnit2 = ResidualConvUnit()

    def forward(self, context, reshape_size, reassembled=None):

        output = context

        if reassembled is not None:
            res = self.resConfUnit1(reassembled)
            output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, size=reshape_size, mode="bilinear", align_corners=True
        )

        output = self.out_conv(output)

        return output


class DPT(nn.Module):
    def __init__(
        self, backbone_pretrained, pretrained_weights=None, image_size=(320, 240)
    ):
        super().__init__()

        self.backbone = create_vit_hybrid_backbone(
            pretrained=backbone_pretrained, size=image_size
        )
        self.scratch = create_scratch()

        self.scratch.refinenet1 = FeatureFusionBlock()
        self.scratch.refinenet2 = FeatureFusionBlock()
        self.scratch.refinenet3 = FeatureFusionBlock()
        self.scratch.refinenet4 = FeatureFusionBlock()

        self.head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        self.scale = 0.000305
        self.shift = 0.1378

        if pretrained_weights is not None:

            loaded_weights = torch.load(
                pretrained_weights,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )

            weights = match_weights(loaded_weights)

            self.load_state_dict(weights)

    def forward(self, x):
        tokens_1, tokens_2, tokens_3, tokens_4 = self.backbone(x)

        token_1_rn = self.scratch.layer1_rn(tokens_1)
        token_2_rn = self.scratch.layer2_rn(tokens_2)
        token_3_rn = self.scratch.layer3_rn(tokens_3)
        token_4_rn = self.scratch.layer4_rn(tokens_4)

        context = self.scratch.refinenet4(token_4_rn, token_3_rn.shape[2:])

        context = self.scratch.refinenet3(
            context, token_2_rn.shape[2:], reassembled=token_3_rn
        )
        context = self.scratch.refinenet2(
            context, token_1_rn.shape[2:], reassembled=token_2_rn
        )
        context = self.scratch.refinenet1(
            context,
            (token_1_rn.shape[2] * 2, token_1_rn.shape[3] * 2),
            reassembled=token_1_rn,
        )

        inv_depth = self.head(context).squeeze(dim=1)

        depth = self.scale * inv_depth + self.shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth
        return depth


if __name__ == "__main__":
    model = DPT(
        pretrained=True,
        pretrained_weights="weights/dpt_hybrid-midas-501f0c75.pt",
        image_size=(256, 256),
    )

    inp = torch.zeros(3, 3, 256, 256) * 2 - 1
    output = model(inp)
    print(output.shape, output.min(), output.max())
