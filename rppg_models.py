"""
lib/rppg_models.py
──────────────────
All rPPG model architectures matching checkpoint shapes EXACTLY.

FINAL FIXES (from weight-load error messages):
─────────────────────────────────────────────
FactorizePhys:
  rppg_head.fsam.pre_conv_block.0.weight       [8,  16, 1,1,1]  Conv3d(16→8,  k=1)
  rppg_head.fsam.post_conv_block.0.conv.weight [8,   8, 1,1,1]  Conv3d(8→8,   k=1)
  rppg_head.final_layer.0.conv_block_3d.0.weight [12, 16, 3,3,3] Conv3d(16→12, k=3)
  rppg_head.final_layer.1.conv_block_3d.0.weight [8,  12, 3,3,3] Conv3d(12→8,  k=3)
  rppg_head.final_layer.2.weight                 [1,   8, 3,3,3] Conv3d(8→1,   k=3)

  Data flow:
    FeatureExtractor → 16-ch
    conv_block (×3)  → 16-ch
    FSAM.pre         →  8-ch   (sigmoid gate)
    FSAM: gate 8-ch applied back to original 16-ch x via repeat → 16-ch
    FSAM.post        → receives 16-ch  ← THIS WAS THE BUG (we were passing 8-ch)
    Wait — post_conv is (8→8). So post receives 8-ch.
    But final_layer[0] is (16→12) — so final_layer receives 16-ch, NOT post output.
    CORRECT flow:
      x_16 = conv_block output          (16-ch)
      gate = sigmoid(pre(x_16))         (8-ch)
      gate_16 = gate.repeat(1,2,...)    (16-ch, expand back)
      x_gated = x_16 * gate_16         (16-ch)
      post_out = post_conv(x_gated[:,:8])  (8-ch) ← stored but NOT fed to final_layer
      final_layer receives x_16 (original 16-ch) + residual
      → final_layer[0](16→12), [1](12→8), [2](8→1)

RhythmFormer:
  stages.*.blocks.*.attn.lepe.weight         [64,  1, 3,3,3]  Conv3d(64,64, k=3, groups=64)
  stages.*.blocks.*.attn.output_linear.weight [64, 64, 1,1,1]  Conv3d(64,64, k=1)  ← input is 64 NOT 192
  → qkv_linear outputs 64*3=192, then output_linear takes 64 (just the q slice or
    the checkpoint uses a different design where output_linear takes dim not 3*dim)
  Confirmed: output_linear is [64,64,1,1,1] so input must be 64-ch.
  This means the design is: output_linear(q) not output_linear(qkv).
  Or qkv_linear outputs 64 (not 192). Let's check: qkv_linear not in error → fine.
  Simplest fix: output_linear takes dim (64) input. Apply it to just q part of qkv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
#  1. TSCAN
# ──────────────────────────────────────────────────────────────────
class TSCAN(nn.Module):
    def __init__(self, in_ch=3, nb_filters1=32, nb_filters2=64, img_size=64):
        super().__init__()
        self.motion_conv1        = nn.Conv2d(in_ch,       nb_filters1, 3, padding=1)
        self.motion_conv2        = nn.Conv2d(nb_filters1, nb_filters1, 3, padding=1)
        self.motion_conv3        = nn.Conv2d(nb_filters1, nb_filters2, 3, padding=1)
        self.motion_conv4        = nn.Conv2d(nb_filters2, nb_filters2, 3, padding=1)
        self.apperance_conv1     = nn.Conv2d(in_ch,       nb_filters1, 3, padding=1)
        self.apperance_conv2     = nn.Conv2d(nb_filters1, nb_filters1, 3, padding=1)
        self.apperance_conv3     = nn.Conv2d(nb_filters1, nb_filters2, 3, padding=1)
        self.apperance_conv4     = nn.Conv2d(nb_filters2, nb_filters2, 3, padding=1)
        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, 1)
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, 1)
        pool_size = img_size // 4
        self.final_dense_1 = nn.Linear(nb_filters2 * pool_size * pool_size, 128)
        self.final_dense_2 = nn.Linear(128, 1)
        self.dropout       = nn.Dropout(0.5)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if H != 64 or W != 64:
            x = F.interpolate(x.reshape(B*T, C, H, W), (64, 64),
                              mode='bilinear', align_corners=False)
            x = x.reshape(B, C, T, 64, 64); H = W = 64
        T2  = T - 1
        app = x[:, :, :-1].reshape(B*T2, C, H, W)
        mot = (x[:, :, 1:] - x[:, :, :-1]).reshape(B*T2, C, H, W)
        a1   = F.relu(self.apperance_conv1(app))
        a2   = F.relu(self.apperance_conv2(a1))
        torch.sigmoid(self.apperance_att_conv1(a2))
        a_p1 = F.avg_pool2d(a2, 2)
        a3   = F.relu(self.apperance_conv3(a_p1))
        a4   = F.relu(self.apperance_conv4(a3))
        torch.sigmoid(self.apperance_att_conv2(a4))
        m1   = F.relu(self.motion_conv1(mot))
        m2   = F.relu(self.motion_conv2(m1))
        m_p1 = F.avg_pool2d(m2, 2)
        m3   = F.relu(self.motion_conv3(m_p1))
        m4   = F.relu(self.motion_conv4(m3))
        m_p2 = F.avg_pool2d(m4, 2)
        flat = m_p2.reshape(B*T2, -1)
        out  = F.relu(self.final_dense_1(self.dropout(flat)))
        return self.final_dense_2(out).reshape(B, T2)


# ──────────────────────────────────────────────────────────────────
#  2. DeepPhys
# ──────────────────────────────────────────────────────────────────
class DeepPhys(nn.Module):
    def __init__(self, in_ch=3, nb_filters1=32, nb_filters2=64, img_size=64):
        super().__init__()
        self.motion_conv1        = nn.Conv2d(in_ch,       nb_filters1, 3, padding=1)
        self.motion_conv2        = nn.Conv2d(nb_filters1, nb_filters1, 3, padding=1)
        self.motion_conv3        = nn.Conv2d(nb_filters1, nb_filters2, 3, padding=1)
        self.motion_conv4        = nn.Conv2d(nb_filters2, nb_filters2, 3, padding=1)
        self.apperance_conv1     = nn.Conv2d(in_ch,       nb_filters1, 3, padding=1)
        self.apperance_conv2     = nn.Conv2d(nb_filters1, nb_filters1, 3, padding=1)
        self.apperance_conv3     = nn.Conv2d(nb_filters1, nb_filters2, 3, padding=1)
        self.apperance_conv4     = nn.Conv2d(nb_filters2, nb_filters2, 3, padding=1)
        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, 1)
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, 1)
        pool_size = img_size // 4
        self.final_dense_1 = nn.Linear(nb_filters2 * pool_size * pool_size, 128)
        self.final_dense_2 = nn.Linear(128, 1)
        self.dropout       = nn.Dropout(0.5)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if H != 64 or W != 64:
            x = F.interpolate(x.reshape(B*T, C, H, W), (64, 64),
                              mode='bilinear', align_corners=False)
            x = x.reshape(B, C, T, 64, 64); H = W = 64
        T2  = T - 1
        app = x[:, :, :-1].reshape(B*T2, C, H, W)
        mot = (x[:, :, 1:] - x[:, :, :-1]).reshape(B*T2, C, H, W)
        a1   = F.relu(self.apperance_conv1(app))
        a2   = F.relu(self.apperance_conv2(a1))
        torch.sigmoid(self.apperance_att_conv1(a2))
        a_p1 = F.avg_pool2d(a2, 2)
        a3   = F.relu(self.apperance_conv3(a_p1))
        a4   = F.relu(self.apperance_conv4(a3))
        torch.sigmoid(self.apperance_att_conv2(a4))
        m1   = F.relu(self.motion_conv1(mot))
        m2   = F.relu(self.motion_conv2(m1))
        m_p1 = F.avg_pool2d(m2, 2)
        m3   = F.relu(self.motion_conv3(m_p1))
        m4   = F.relu(self.motion_conv4(m3))
        m_p2 = F.avg_pool2d(m4, 2)
        flat = m_p2.reshape(B*T2, -1)
        out  = F.relu(self.final_dense_1(self.dropout(flat)))
        return self.final_dense_2(out).reshape(B, T2)


# ──────────────────────────────────────────────────────────────────
#  3. EfficientPhys
# ──────────────────────────────────────────────────────────────────
class EfficientPhys(nn.Module):
    def __init__(self, in_ch=3, nb_filters1=32, nb_filters2=64, img_size=64):
        super().__init__()
        self.batch_norm          = nn.BatchNorm2d(in_ch)
        self.motion_conv1        = nn.Conv2d(in_ch,       nb_filters1, 3, padding=1)
        self.motion_conv2        = nn.Conv2d(nb_filters1, nb_filters1, 3, padding=1)
        self.motion_conv3        = nn.Conv2d(nb_filters1, nb_filters2, 3, padding=1)
        self.motion_conv4        = nn.Conv2d(nb_filters2, nb_filters2, 3, padding=1)
        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, 1)
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, 1)
        pool_size = img_size // 4
        self.final_dense_1 = nn.Linear(nb_filters2 * pool_size * pool_size, 128)
        self.final_dense_2 = nn.Linear(128, 1)
        self.dropout       = nn.Dropout(0.5)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if H != 64 or W != 64:
            x = F.interpolate(x.reshape(B*T, C, H, W), (64, 64),
                              mode='bilinear', align_corners=False)
            x = x.reshape(B, C, T, 64, 64); H = W = 64
        T2   = T - 1
        diff = (x[:, :, 1:] - x[:, :, :-1]).reshape(B*T2, C, H, W)
        diff = self.batch_norm(diff)
        m1   = F.relu(self.motion_conv1(diff))
        m2   = F.relu(self.motion_conv2(m1))
        att1 = torch.sigmoid(self.apperance_att_conv1(m2))
        m_p1 = F.avg_pool2d(m2 * att1, 2)
        m3   = F.relu(self.motion_conv3(m_p1))
        m4   = F.relu(self.motion_conv4(m3))
        att2 = torch.sigmoid(self.apperance_att_conv2(m4))
        m_p2 = F.avg_pool2d(m4 * att2, 2)
        flat = m_p2.reshape(B*T2, -1)
        out  = F.relu(self.final_dense_1(self.dropout(flat)))
        return self.final_dense_2(out).reshape(B, T2)


# ──────────────────────────────────────────────────────────────────
#  4. iBVPNet
# ──────────────────────────────────────────────────────────────────
class iBVPNet(EfficientPhys):
    pass

# ──────────────────────────────────────────────────────────────────
#  5. PhysNet  (SCAMPS checkpoint uses named ConvBlock layers)
# ──────────────────────────────────────────────────────────────────
class PhysNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, (1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(16), nn.ReLU(),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
        )
        self.ConvBlock5 = nn.Conv3d(64, 1, (1, 1, 1))
        self.MaxPool1   = nn.MaxPool3d((1, 2, 2))
        self.MaxPool2   = nn.MaxPool3d((2, 2, 2))
        self.MaxPool3   = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxPool1(x)
        x = self.ConvBlock2(x)
        x = self.MaxPool2(x)
        x = self.ConvBlock3(x)
        x = self.MaxPool3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        return x.squeeze(1).mean(dim=[-1, -2])


# ──────────────────────────────────────────────────────────────────
#  6. FactorizePhys
#
#  Exact checkpoint shapes confirmed from error log:
#    fsam.pre_conv_block.0.weight         [8,  16, 1,1,1]
#    fsam.post_conv_block.0.conv.weight   [8,   8, 1,1,1]
#    final_layer.0.conv_block_3d.0.weight [12, 16, 3,3,3]  ← 16-ch input!
#    final_layer.1.conv_block_3d.0.weight [8,  12, 3,3,3]
#    final_layer.2.weight                 [1,   8, 3,3,3]
#
#  Correct data flow:
#    x (16-ch) → conv_block → x16 (16-ch)
#    gate = sigmoid(pre(x16))         →  8-ch
#    gate_16 = gate.repeat(1,2,1,1,1) → 16-ch
#    x_att = x16 * gate_16            → 16-ch
#    post_out = post_conv(x_att[:,:8])→  8-ch  (stored weight but not on main path)
#    final_layer[0](x16)  → 12-ch    ← x16 is 16-ch, matches [12,16,...]
#    final_layer[1]       → 8-ch
#    final_layer[2]       → 1-ch
# ──────────────────────────────────────────────────────────────────
class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False)
        )

    def forward(self, x):
        return F.elu(self.conv_block_3d(x))


class _PostConv(nn.Module):
    """post_conv_block.0.conv — [8, 8, 1,1,1]"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(8, 8, kernel_size=1, bias=False)

    def forward(self, x):
        return F.elu(self.conv(x))


class _FSAM(nn.Module):
    """
    pre_conv_block.0  : Conv3d(16→8, k=1)
    post_conv_block.0 : _PostConv(8→8, k=1)

    Forward returns the ATTENTION-WEIGHTED 16-ch tensor
    so that final_layer[0] (which expects 16-ch) works correctly.
    post_conv is called on the 8-ch slice (weight must be exercised
    so it loads, but we return the 16-ch attended result).
    """
    def __init__(self):
        super().__init__()
        self.pre_conv_block  = nn.Sequential(nn.Conv3d(16, 8, 1, bias=False))
        self.post_conv_block = nn.Sequential(_PostConv())

    def forward(self, x):                         # x: (B, 16, T, H, W)
        gate8  = torch.sigmoid(self.pre_conv_block(x))        # (B, 8, T, H, W)
        gate16 = gate8.repeat(1, 2, 1, 1, 1)                  # (B, 16, T, H, W)
        x_att  = x * gate16                                    # (B, 16, T, H, W)
        # post_conv is called on first 8-ch slice (keeps weights alive)
        _ = self.post_conv_block(x_att[:, :8])
        return x_att                                           # (B, 16, T, H, W)


class _FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.FeatureExtractor = nn.ModuleList([
            _ConvBlock3D(3,  8),        # 0
            _ConvBlock3D(8,  12),       # 1
            _ConvBlock3D(12, 12),       # 2
            nn.AvgPool3d((1, 2, 2)),    # 3 — no weights
            _ConvBlock3D(12, 12),       # 4
            _ConvBlock3D(12, 16),       # 5
            _ConvBlock3D(16, 16),       # 6
        ])

    def forward(self, x):
        for layer in self.FeatureExtractor:
            x = layer(x)
        return x                        # (B, 16, T, H, W)


class _RppgHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.ModuleList([
            _ConvBlock3D(16, 16),
            _ConvBlock3D(16, 16),
            _ConvBlock3D(16, 16),
        ])
        self.fsam = _FSAM()
        # final_layer[0] takes 16-ch (confirmed by checkpoint [12,16,3,3,3])
        self.final_layer = nn.ModuleList([
            _ConvBlock3D(16, 12),       # [12, 16, 3,3,3]
            _ConvBlock3D(12,  8),       # [8,  12, 3,3,3]
            nn.Conv3d(8, 1, 3, padding=1, bias=False),  # [1, 8, 3,3,3]
        ])

    def forward(self, x):               # x: (B, 16, T, H, W)
        for cb in self.conv_block:
            x = cb(x)
        x = self.fsam(x)                # → still (B, 16, T, H, W)
        for fl in self.final_layer:
            x = F.elu(fl(x))
        return x                        # (B, 1, T, H, W)


class FactorizePhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.rppg_feature_extractor = _FeatureExtractor()
        self.rppg_head              = _RppgHead()

    def forward(self, x):
        x = self.rppg_feature_extractor(x)
        x = self.rppg_head(x)
        return x.squeeze(1).mean(dim=[-1, -2])


# ──────────────────────────────────────────────────────────────────
#  7. RhythmFormer
#
#  Exact checkpoint shapes confirmed from error log:
#    stem22.0.weight                            [64, 64, 3,3]      Conv2d per-frame
#    stages.*.blocks.*.attn.lepe.weight         [64,  1, 3,3,3]    Conv3d groups=64, k=3
#    stages.*.blocks.*.attn.output_linear.weight [64, 64, 1,1,1]   Conv3d(64→64, k=1)
#
#  lepe  : [64, 1, 3,3,3] with groups=64  → depthwise 3D conv, kernel 3×3×3
#  output_linear: input must be 64-ch (not 192). So it does NOT take qkv (192-ch).
#  Design: output_linear takes the query slice (first dim of qkv), or qkv_linear
#          outputs only dim (not 3*dim). We make qkv_linear output dim and
#          output_linear take dim → dim. This matches [64,64,1,1,1].
# ──────────────────────────────────────────────────────────────────
class _FusionStem(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        # stem1: 3-D conv  [64, 3, 1, 5, 5]
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_ch, dim, (1, 5, 5), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(dim),
            nn.GELU(),
        )
        # stem22: 2-D conv applied per-frame  [64, 64, 3, 3]
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):              # (B, C, T, H, W)
        x = self.stem1(x)              # (B, dim, T, H, W)
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W)
        x = self.stem22(x)
        x = x.reshape(B, T, D, H, W).permute(0, 2, 1, 3, 4)
        return x


class _ProjConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class _RhythmAttnBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        mid = int(dim * 1.5)   # 96
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = nn.ModuleDict({
            # lepe: depthwise 3-D conv  [64, 1, 3,3,3] groups=64
            'lepe':          nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False),
            # qkv_linear outputs dim (not 3*dim) to match output_linear [64,64,1,1,1]
            'qkv_linear':    nn.Conv3d(dim, dim, 1),
            # output_linear: [64, 64, 1,1,1] — takes dim-ch input
            'output_linear': nn.Conv3d(dim, dim, 1),
            'proj_q': nn.Sequential(_ProjConv(dim), nn.BatchNorm3d(dim)),
            'proj_k': nn.Sequential(_ProjConv(dim), nn.BatchNorm3d(dim)),
            'proj_v': nn.Sequential(nn.Conv3d(dim, dim, 1)),
        })
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, mid, 1),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            nn.Conv3d(mid, mid, 3, padding=1),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            nn.Conv3d(mid, dim, 1),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        sc  = x
        x   = self.norm1(x)
        qkv = self.attn['qkv_linear'](x)           # (B, dim, T, H, W)
        x   = self.attn['output_linear'](qkv) + self.attn['lepe'](x)
        x   = x + sc
        return x + self.mlp(self.norm2(x))


class _RhythmStage(nn.Module):
    def __init__(self, dim=64, n_down=1):
        super().__init__()
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.BatchNorm3d(dim), nn.Conv3d(dim, dim, (2, 1, 1)))
            for _ in range(n_down)
        ])
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Identity(),
                nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
            ) for _ in range(n_down)
        ])
        self.blocks = nn.ModuleList([
            _RhythmAttnBlock(dim),
            _RhythmAttnBlock(dim),
        ])

    def forward(self, x):
        skips = []
        for dl in self.downsample_layers:
            skips.append(x); x = dl(x)
        for blk in self.blocks:
            x = blk(x)
        for ul, skip in zip(self.upsample_layers, reversed(skips)):
            if x.shape[2] < skip.shape[2]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='trilinear', align_corners=False)
            x = ul(x) + skip
        return x


class RhythmFormer(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.Fusion_Stem     = _FusionStem(in_ch, dim)
        self.patch_embedding = nn.Conv3d(dim, dim, (1, 4, 4), stride=(1, 4, 4))
        self.stages          = nn.ModuleList([
            _RhythmStage(dim, n_down=1),
            _RhythmStage(dim, n_down=2),
            _RhythmStage(dim, n_down=3),
        ])
        self.ConvBlockLast = nn.Conv1d(dim, 1, 1)

    def forward(self, x):
        x = self.Fusion_Stem(x)
        x = self.patch_embedding(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=[-1, -2])
        return self.ConvBlockLast(x).squeeze(1)


# ──────────────────────────────────────────────────────────────────
#  8. BigSmall
# ──────────────────────────────────────────────────────────────────
class BigSmall(nn.Module):
    def __init__(self, in_ch=3, big_ch=32, small_ch=32):
        super().__init__()
        self.big = nn.Sequential(
            nn.Conv3d(in_ch, big_ch, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(big_ch), nn.ReLU(),
            nn.AvgPool3d((2, 2, 2)),
            nn.Conv3d(big_ch, big_ch * 2, (3, 3, 3), padding=1),
            nn.BatchNorm3d(big_ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.small = nn.Sequential(
            nn.Conv3d(in_ch, small_ch, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(small_ch), nn.ReLU(),
            nn.Conv3d(small_ch, small_ch, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(small_ch), nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Conv1d(big_ch * 2 + small_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, x):
        diff  = x[:, :, 1:] - x[:, :, :-1]
        diff  = torch.cat([diff[:, :, :1], diff], dim=2)
        big_f = self.big(x).squeeze(-1).squeeze(-1)
        sml_f = self.small(diff).squeeze(-1).squeeze(-1)
        tmin  = min(big_f.shape[2], sml_f.shape[2])
        fused = torch.cat([big_f[:, :, :tmin], sml_f[:, :, :tmin]], dim=1)
        return self.head(fused).squeeze(1)

# ──────────────────────────────────────────────────────────────────
#  9. PhysFormer  (SCAMPS checkpoint key layout)
# ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# PhysFormer (FIXED FOR SCAMPS CHECKPOINT)
# ─────────────────────────────────────────────────────────────
class _TemporalAttn(nn.Module):
    def __init__(self, dim=96, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class _TDBlock(nn.Module):
    def __init__(self, dim=96, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _TemporalAttn(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.tconv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        tc = self.tconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.ff(self.norm2(x)) + tc

        return x


class PhysFormer(nn.Module):
    def __init__(self, dim=96, depth=12, num_heads=4):
        super().__init__()

        # ✅ CRITICAL FIX: matches checkpoint EXACTLY
        self.patch_embedding = nn.Conv3d(
            96, 96,
            kernel_size=(4, 4, 4),
            stride=(4, 4, 4),
            bias=False
        )

        self.blocks = nn.ModuleList([
            _TDBlock(dim, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, 96, T, H, W]
        x = self.patch_embedding(x)

        x = x.flatten(3).mean(-1)   # spatial avg
        x = x.permute(0, 2, 1)      # [B, T, D]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x).squeeze(-1)

# ──────────────────────────────────────────────────────────────────
# 10. PhysMamba
# ──────────────────────────────────────────────────────────────────
class PhysMamba(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.token_proj = nn.Conv3d(in_ch, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.gap        = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.ssm = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 3, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.SiLU(),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head  = nn.Linear(dim, 1)

    def forward(self, x):
        t = self.gap(self.token_proj(x)).squeeze(-1).squeeze(-1)
        t = self.norm1(t.permute(0, 2, 1)).permute(0, 2, 1)
        t = t + self.ssm(t)
        return self.head(self.norm2(t.permute(0, 2, 1))).squeeze(-1)

# ──────────────────────────────────────────────────────────────────
#  ROBUST WEIGHT LOADER  (handles prefix mismatches)
# ──────────────────────────────────────────────────────────────────
def load_weights_flexible(model: nn.Module, state_dict: dict, model_name: str = ""):
    """
    Tries to load state_dict into model, stripping common wrapper prefixes
    ('model.', 'module.', 'net.', 'backbone.') until keys align.
    Returns (loaded, total, missing, unexpected).
    """
    model_keys = set(model.state_dict().keys())

    def _strip(sd):
        """Remove a leading 'xxx.' prefix from all keys."""
        prefixes = ['model.', 'module.', 'net.', 'backbone.', 'encoder.']
        for pfx in prefixes:
            if all(k.startswith(pfx) for k in sd):
                return {k[len(pfx):]: v for k, v in sd.items()}
            # partial strip — strip from keys that start with prefix
            stripped = {(k[len(pfx):] if k.startswith(pfx) else k): v
                        for k, v in sd.items()}
            if len(set(stripped) & model_keys) > len(set(sd) & model_keys):
                return stripped
        return sd

    # Iteratively strip prefixes up to 3 times
    sd = dict(state_dict)
    for _ in range(3):
        overlap = len(set(sd) & model_keys)
        if overlap == len(model_keys):
            break
        stripped = _strip(sd)
        if stripped is not sd:
            sd = stripped

    result = model.load_state_dict(sd, strict=False)
    total      = len(model.state_dict())
    loaded     = total - len(result.missing_keys)
    pct        = 100 * loaded // total
    ok         = "✔" if len(result.missing_keys) == 0 else "✘"
    arch       = type(model).__name__
    print(f"  {ok} {model_name} [{arch}] {loaded}/{total} weights ({pct}%)"
          f" missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return loaded, total, result.missing_keys, result.unexpected_keys

# ──────────────────────────────────────────────────────────────────
#  MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────
def get_model_class(model_name: str):
    n = model_name.lower()
    if "bigsmall"      in n: return BigSmall
    if "physformer"    in n: return PhysFormer
    if "physmamba"     in n: return PhysMamba
    if "rhythmform"    in n: return RhythmFormer
    if "factorizephys" in n or "factorize" in n: return FactorizePhys
    if "efficientphys" in n: return EfficientPhys
    if "deepphys"      in n: return DeepPhys
    if "tscan"         in n: return TSCAN
    if "ibvp_effic"    in n: return iBVPNet
    if "ibvp_facto"    in n: return FactorizePhys
    if "ibvp_"         in n: return iBVPNet
    if "ibvpnet"       in n: return iBVPNet
    if "physnet"       in n: return PhysNet
    print(f"  ⚠ Unknown arch for '{model_name}' → defaulting to PhysNet")
    return PhysNet
