from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *
import torch.nn.functional as F
import torch

class Resize():
    def __init__(self, size=(3, 224, 224), mode="bilinear"):
        """

        :param size:
        :param mode: bilinear for image, nearest for label
        """
        self.mode = mode
        self.size = size
        self._pad_mode_convert = {'reflection': 'reflect', 'zeros': 'constant', 'border': 'replicate'}

    def resize_crop(self, x):
        C, H, W = x.size()
        target_r, target_c = self.size[1:]
        ratio = min(H / target_r, W / target_c)
        return int(round(H / ratio)), int(round(W / ratio))

    def _crop_pad_default(self, x, size, padding_mode='reflection', row_pct=0.5, col_pct=0.5):
        "Crop and pad tfm - `row_pct`,`col_pct` sets focal point."
        padding_mode = self._pad_mode_convert[padding_mode]
        if x.shape[1:] == torch.Size(size): return x
        rows, cols = size
        if x.size(1) < rows or x.size(2) < cols:
            row_pad = max((rows - x.size(1) + 1) // 2, 0)
            col_pad = max((cols - x.size(2) + 1) // 2, 0)
            x = F.pad(x[None], (col_pad, col_pad, row_pad, row_pad), mode=padding_mode)[0]
        row = int((x.size(1) - rows + 1) * row_pct)
        col = int((x.size(2) - cols + 1) * col_pct)
        x = x[:, row:row + rows, col:col + cols]
        return x.contiguous()  # without this, get NaN later - don't know why

    def _affine_grid(self, size):
        size = ((1,) + size)
        N, C, H, W = size
        grid = torch.FloatTensor(N, H, W, 2)
        linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1.])
        grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1.])
        grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
        return grid

    def _grid_sample(self, x, flow, padding_mode='reflection', remove_out=True):
        coords = flow.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
        if self.mode == 'bilinear':  # hack to get smoother downwards resampling
            mn, mx = coords.min(), coords.max()
            # max amount we're affine zooming by (>1 means zooming in)
            z = 1 / (mx - mn).item() * 2
            # amount we're resizing by, with 100% extra margin
            d = min(x.shape[1] / coords.shape[1], x.shape[2] / coords.shape[2]) / 2
            # If we're resizing up by >200%, and we're zooming less than that, interpolate first
            if d > 1 and d > z: x = F.interpolate(x[None], scale_factor=1 / d, mode='area')[0]
        kwargs = {'mode': self.mode, 'padding_mode': padding_mode}
        if torch.__version__ > "1.2.0": kwargs['align_corners'] = True
        return F.grid_sample(x[None], coords, **kwargs)[0]

    def transform(self, x):
        # x should be Tensor with shape [C, H, W]
        size = self.resize_crop(x.float())

        flow = self._affine_grid((x.size(0),) + size)
        tmp_x = self._grid_sample(x, flow)
        return self._crop_pad_default(tmp_x, self.size[1:])

    def __call__(self, x):
        return self.transform(x)
