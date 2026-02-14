"""
    BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment, CVPR 2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Conditional mmcv imports ----------
# Use real deformable convolutions when mmcv is available (CUDA/Linux),
# fall back to standard Conv2d on macOS/MPS where mmcv can't compile.
try:
    from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
    from mmcv.cnn import constant_init
    _MMCV_AVAILABLE = True
    print("[feat_prop] Using mmcv deformable convolutions (CUDA)")
except ImportError:
    _MMCV_AVAILABLE = False
    print("[feat_prop] mmcv not available — using standard Conv2d fallback")

    class ModulatedDeformConv2d(nn.Module):
        """Fallback replacement for mmcv.ops.ModulatedDeformConv2d.
        Uses standard Conv2d instead of deformable convolution."""
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, groups=1,
                     deform_groups=1, bias=True):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            self.groups = groups
            self.deform_groups = deform_groups
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.bias = None
            nn.init.kaiming_uniform_(self.weight)

        def forward(self, x, offset=None, mask=None):
            return F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    def modulated_deform_conv2d(x, offset, mask, weight, bias,
                                stride, padding, dilation, groups, deform_groups):
        """Fallback: ignore offset/mask, use standard conv2d."""
        _stride = stride if isinstance(stride, tuple) else (stride, stride)
        _padding = padding if isinstance(padding, tuple) else (padding, padding)
        _dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        return F.conv2d(x, weight, bias, _stride, _padding, _dilation, groups)

    def constant_init(module, val, bias=0):
        """Replacement for mmcv.cnn.constant_init."""
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
# ---------- end conditional imports ----------

from E2FGVI.model.modules.flow_comp import flow_warp


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x, flows_backward, flows_forward):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []

            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        flow_n2 = flow_n1 + flow_warp(
                            flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2,
                                            flow_n2.permute(0, 2, 3, 1))

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                               flow_n1,
                                                               flow_n2)

                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', module_name]
                ] + [feat_prop]

                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x
