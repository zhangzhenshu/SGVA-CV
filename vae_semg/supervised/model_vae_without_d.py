import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(z_dim, 240, bias=False), nn.BatchNorm1d(240), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(24, 36, kernel_size=(4, 11), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(36), nn.ReLU())
        self.up2 = nn.Upsample([4, 40])
        self.de2 = nn.Sequential(nn.ConvTranspose2d(36, 48, kernel_size=(5, 13), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(48), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 24, 1, 10)
        h = self.up1(h)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)

        return loc_img


# Encoders
class qz(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qz, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),  # 1024*24*1*10
        )

        self.fc11 = nn.Sequential(nn.Linear(240, z_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 24*1*10)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(z_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(z_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_dim = args.z_dim
        self.d_dim = args.d_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        self.qz = qz(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        # self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.z_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        # self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta = args.beta

        self.cuda()

    def forward(self, d, x, y):
        # Encode
        z_q_loc, z_q_scale = self.qz(x)

        # Reparameterization trick
        qz = dist.Normal(z_q_loc, z_q_scale)
        z_q = qz.rsample()

        # Decode
        x_recon = self.px(z_q)

        # Priors
        z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.z_dim).cuda(),\
                               torch.ones(z_q.size()[0], self.z_dim).cuda()
        pz = dist.Normal(z_p_loc, z_p_scale)

        # Auxiliary losses
        # d_hat = self.qd(z_q)
        y_hat = self.qy(z_q)

        return x_recon,  y_hat, qz, pz, z_q

    def loss_function(self, d, x, y):
        x_recon,  y_hat, qz, pz, z_q = self.forward(d, x, y)
        x_recon = x_recon.permute(0, 3, 2, 1)
        x_recon = x_recon.reshape(-1,  48)
        x_target = (x.view(-1)).long()
        CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

        KL_z = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        # _, d_target = d.max(dim=1)
        # CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

        _, y_target = y.max(dim=1)
        CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

        return CE_x - self.beta * KL_z + self.aux_loss_multiplier_y * CE_y, CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            z_q_loc, z_q_scale = self.qz(x)
            z = z_q_loc
            # alpha = F.softmax(self.qd(z), dim=1)
            #
            # # get the index (digit) that corresponds to
            # # the maximum predicted class probability
            # res, ind = torch.topk(alpha, 1)
            #
            # # convert the digit(s) to one-hot tensor(s)
            # d = x.new_zeros(alpha.size())
            # d = d.scatter_(1, ind, 1.0)

            alpha = F.softmax(self.qy(z), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

        return  y