import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist


### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

'''
SGVA model
'''
# Decoders
class px(nn.Module):
    '''
    Decoder of SGVA
    params:
        zd_dim: size of latent space of subject label
        zx_dim: size of latent space of input signal
        zy_dim: size of latent space of gesture label
    return:
        reconstruction of signal
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 240, bias=False), nn.BatchNorm1d(240), nn.ReLU())
        # self.up1 = nn.Upsample(8)
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

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)  # 1024*240
        h = h.view(-1, 24, 1, 10)  # 1024*24*1*10
        h = self.de1(h)  # 1024*36*4*20
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)

        return loc_img  # 1024*48*8*52


class pzd(nn.Module):
    '''
    Mapping subject label to a distribution in latent space zd
    params:
        d_dim: number of source domain
        zd_dim: size of latent space of subject label
    return:
        zd_loc: Mean of the latent space distribution
        zd_scale: Standard deviation of the latent space distribution
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):  # 1024*6
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        return zd_loc, zd_scale


class pzy(nn.Module):
    '''
    Mapping gesture label to a distribution in latent space zy
    params:
        y_dim: number of source domain
        zy_dim: size of latent space of subject label
    return:
        zy_loc: Mean of the latent space distribution
        zy_scale: Standard deviation of the latent space distribution
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    '''
    Subject encoder of SGVA
    params:
        zd_dim: size of latent space of subject label
    return:
        zd_loc: Mean of the latent space distribution
        zd_scale: Standard deviation of the latent space distribution
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()
        # 1024*1*8*52
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )

        self.fc11 = nn.Sequential(nn.Linear(240, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)  # 1024*24*1*10

        h = h.view(-1, 24 * 1 * 10)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


class qzx(nn.Module):
    '''
    Sample encoder of SGVA
    params:
        zx_dim: size of latent space of subject label
    return:
        zx_loc: Mean of the latent space distribution
        zx_scale: Standard deviation of the latent space distribution
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )

        self.fc11 = nn.Sequential(nn.Linear(240, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zx_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)  # 初始化两个卷积层的权重
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)  # 1024*24*1*10
        h = h.view(-1, 24 * 1 * 10)
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):
    '''
    Gesture encoder of SGVA-CV
    params:
        zy_dim: size of latent space of subject label
    return:
        zy_loc: Mean of the latent space distribution
        zy_scale: Standard deviation of the latent space distribution
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )

        self.fc11 = nn.Sequential(nn.Linear(240, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 240)
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7

        return zy_loc, zy_scale


# Auxiliary tasks
class qd(nn.Module):
    '''
    Subject classifier
    params:
        d_dim:
        zd_dim: size of latent space of subject label
    return:
        loc_d: predicted subject label
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)
        # self.dense_layer = nn.Sequential(
        #     nn.Linear(zd_dim, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(64),
        #     nn.Dropout(0.3),
        #
        #     nn.Linear(64, d_dim)
        # )

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.dense_layer.weight)
        # self.dense_layer.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    '''
    Gesture classifier
    params:
        y_dim:
        zy_dim: size of latent space of gesture label
    return:
        loc_y: predicted gesture label
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)
        # self.dense_layer = nn.Sequential(
        #     nn.Linear(zd_dim, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(64),
        #     nn.Dropout(0.3),
        #
        #     nn.Linear(64, y_dim)
        # )

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.dense_layer.weight)
        # self.dense_layer.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class DIVA(nn.Module):
    def __init__(self, args):
        super(DIVA, self).__init__()
        self.zd_dim = args.zd_dim  # size of latent space
        self.zx_dim = args.zx_dim
        self.zy_dim = args.zy_dim
        self.d_dim = args.d_dim  # number of domain
        self.x_dim = args.x_dim  # input size after flattening
        self.y_dim = args.y_dim  # number of classes

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.cuda()

    def forward(self, d, x, y):
        """
        x_recon, d_hat, y_hat: 重建后的值
        zd_q, zx_q, zy_q:中间隐变量
        """
        # x -> zd zx zy ->x_recon(inference model -> generative model)
        # 1 Encode : getting parameters of Gaussian distribution
        zd_q_loc, zd_q_scale = self.qzd(x)  # 1024*1*8*52 -> 1024*64
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)  # 1024*64
        zy_q_loc, zy_q_scale = self.qzy(x)  # 1024*64

        # 2 Reparameterization trick  : Creating Gaussian distribution
        qzd = dist.Normal(zd_q_loc, zd_q_scale)  # 1024*64
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)  # 1024*64
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)  # 1024*64
        zy_q = qzy.rsample()

        # --------------3 Decode----------------#
        x_recon = self.px(zd_q, zx_q, zy_q)  # 1024*48*8*52

        # d,y ->zd,zy -> d_hat,y_hat    (generative model ->inference model)
        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # --------Auxiliary losses-----------#
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # Do standard forward pass for everything not involving y
            zd_q_loc, zd_q_scale = self.qzd(x)
            if self.zx_dim != 0:
                zx_q_loc, zx_q_scale = self.qzx(x)
            zy_q_loc, zy_q_scale = self.qzy(x)

            qzd = dist.Normal(zd_q_loc, zd_q_scale)
            zd_q = qzd.rsample()
            if self.zx_dim != 0:
                qzx = dist.Normal(zx_q_loc, zx_q_scale)
                zx_q = qzx.rsample()
            else:
                zx_q = None
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            zy_q = qzy.rsample()

            zd_p_loc, zd_p_scale = self.pzd(d)
            if self.zx_dim != 0:
                zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                    torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()

            pzd = dist.Normal(zd_p_loc, zd_p_scale)

            if self.zx_dim != 0:
                pzx = dist.Normal(zx_p_loc, zx_p_scale)
            else:
                pzx = None

            d_hat = self.qd(zd_q)

            x_recon = self.px(zd_q, zx_q, zy_q)

            x_recon = x_recon.view(-1, 256)
            x_target = (x.view(-1) * 255).long()
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(10)
            y_onehot = y_onehot.repeat(1, 100)
            y_onehot = y_onehot.view(1000, 10).cuda()

            zy_q = zy_q.repeat(10, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(10, 1), zy_q_scale.repeat(10, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)

            # Reparameterization trick
            pzy = dist.Normal(zy_p_loc, zy_p_scale)

            # Auxiliary losses
            y_hat = self.qy(zy_q)

            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1)
            qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)

            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)

            prior_y = torch.tensor(1 / 10).cuda()
            prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)

            return CE_x \
                - self.beta_d * zd_p_minus_zd_q \
                - self.beta_x * KL_zx \
                - self.beta_y * marginal_zy_p_minus_zy_q \
                - marginal_prior_y_minus_qy \
                + self.aux_loss_multiplier_d * CE_d

        else:  # supervised
            # reconstruction loss
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)
            x_recon = x_recon.permute(0, 2, 3, 1)
            x_recon = x_recon.reshape(-1, 48)
            # x_recon = x_recon.view(-1, 48)
            x_target = (x.permute(0, 1, 3, 2).reshape(-1)).long()
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

            # KL Divergence
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))  # -KL
            else:
                KL_zx = 0
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

            # classification loss of subject and gesture
            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y, \
                CE_y, zd_q, zy_q, zx_q, d_target, y_target

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)  # 返回最大值和概率

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

        return d, y

    def classifier_for_tsne(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)  # 返回最大值和概率

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

        return d, y, zy
