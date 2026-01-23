import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# ------------------------------------------
#           CycleGAN Generator (ResNet)
# ------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_filters=64, n_blocks=9):
        super(Generator, self).__init__()
        # Initial Conv
        model = [
            nn.Conv2d(input_nc, n_filters, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(True)
        ]
        # Downsampling
        curr_dim = n_filters
        n_down = 2
        for i in range(n_down):
            in_dim = curr_dim
            out_dim = curr_dim * 2
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(True)
            ]
            curr_dim = out_dim
        # Resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(curr_dim)]
        # Upsampling
        for i in range(n_down):
            in_dim = curr_dim
            out_dim = curr_dim // 2
            model += [
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(True)
            ]
            curr_dim = out_dim
        # Output Layer
        model += [
            nn.Conv2d(curr_dim, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------
#       CycleGAN Discriminator (PatchGAN)
# ------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc=1, n_filters=64, n_layers=4):
        super(Discriminator, self).__init__()
        sequence = [
            nn.Conv2d(input_nc, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        nf = n_filters
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, True)
            ]
        # final layer
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]
        # output 1-channel prediction map
        sequence += [nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


# ------------------------------
# 定义 FSeSim 损失模块
# ------------------------------
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        #for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return out

class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids

class FSeSimLoss(nn.Module):
    """
    Fixed Self-Similarity Loss.
    在 relu3_1 和 relu4_1 级别，随机采 ns=256 个 query，
    每个 query 再随机选 np=patch_size*patch_size 个 reference，
    对 query / reference 做 L2 归一化后计算内积（cosine similarity），
    最后用 L1 距离聚合。
    """
    def __init__(self, device, layers=('relu3_1','relu4_1'),
                 patch_nums=256, patch_size=32, norm=True):
        super().__init__()
        # VGG16 特征提取器（只输出所有 relu* 层）
        self.vgg = VGG16().to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad = False

        # PatchSim 用来计算每层的自相似图
        self.patch_sim = PatchSim(patch_nums=patch_nums,
                                  patch_size=patch_size,
                                  norm=norm)
        self.layers = layers
        self.device = device

    def forward(self, x, y):
        """
        x, y: [B,1,H,W], in [-1,1]
        returns: scalar FSeSim loss
        """
        # 复制到 3 通道并归一化到 VGG 要求的均值/方差
        x_rgb = x.repeat(1,3,1,1)
        y_rgb = y.repeat(1,3,1,1)

        # 提取所有 relu 特征
        feats_x = self.vgg(x_rgb)  # dict of {'relu1_2', ..., 'relu5_3'}
        feats_y = self.vgg(y_rgb)

        losses = []
        for layer in self.layers:
            fx = feats_x[layer]  # [B,C,Hf,Wf]
            fy = feats_y[layer]

            # patch_sim: 返回 (sim_map, patch_ids), sim_map shape = [B, patch_nums, patch_size*patch_size]
            sim_x, patch_ids = self.patch_sim(fx)
            sim_y, _         = self.patch_sim(fy, patch_ids=patch_ids)

            # sim_x / sim_y 已经是 L2 归一化后的 query·key
            # 用 L1 距离来衡量两者的差异
            losses.append(torch.abs(sim_x - sim_y).mean())

        # 各层平均
        return sum(losses) / len(losses)


if __name__ == '__main__':
    G = Generator(input_nc=1, output_nc=1).cuda()
    x = torch.randn(2,1,256,256).cuda()
    y = G(x)
    print('G out:', y.shape)  # -> [2,1,256,256]
    D = Discriminator(input_nc=1).cuda()
    d = D(y)
    print('D out:', d.shape)  # e.g. [2,1,31,31]
