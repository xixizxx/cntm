import torch
import torch.nn.functional as F 
import numpy as np
# from datetime import datetime
from .utils.plot import get_pos_neg_bows

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temperature_scl = 0.7

class CNTM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize,
                 theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5, cl=0, tau=0):
        super(CNTM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)

        # CL Network
        if cl == 1:
            self.CL_Net = CL_Net(self.vocab_size, tau).to(device)
        
        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta, self.alphas.weight, self.rho.weight

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, bows, normalized_bows, targets, CL=0, label=0, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta, t_emb, w_emb = self.get_beta()

        ## contrastive learning
        cl_loss = torch.tensor(0).to(device)
        if CL == 1:
            if label == 0: # unsupervised contrastive learning
                # print("=== get_pos_neg_bows begin: " + str(datetime.now()))
                pos_bows, neg_bows, real_topics = get_pos_neg_bows(bows, theta, t_emb)
                # print("!!!! get_pos_neg_bows end: " + str(datetime.now()))
                if neg_bows is not None and real_topics is not None:
                    cl_loss = self.CL_Net(real_topics, bows, neg_bows)
                    # print("++++ CL_Net end: " + str(datetime.now()))
            else: # supervised contrastive learning
                pos_bows, neg_bows, real_topics = get_pos_neg_bows(bows, theta, t_emb, targets)
                if pos_bows is not None and real_topics is not None:
                    features = torch.cat([bows.unsqueeze(1), pos_bows.unsqueeze(1)], dim=1)
                    cl_loss = self.get_sc_loss(features, targets)

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta, cl_loss

    # supervised contrastive learning
    def get_sc_loss(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # here mask is of shape [bsz, bsz] and is one for one for [i,j] where label[i]=label[j]
            mask = torch.eq(labels, labels.T).float().to(device)

        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number of positives per sample

        # contrast_features separates the features of different views of the samples and puts them in rows, so features of
        # shape of [50, 2, 128] becomes [100, 128]. we do this to be to calculate dot-product between each two views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits - calculates the dot product of every two vectors divided by temperature = 0.7
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature_scl)

        # for numerical stability  (some kind of normalization!)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask as much as number of positives per sample
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        eps = 1e-30
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        loss = -  mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# unsupervised contrastive learning by Deep InfoMax Loss
class CL_Net(torch.nn.Module):

    def __init__(self, vocab_size, tau, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()

        self.tau = tau

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.e_dim = vocab_size

        self.out_size = 300

        self.device = torch.cuda.current_device()

        self.opt = None

        self.set_models()

    def set_models(self):
        self.D = Discriminator(M_channels=self.e_dim, E_size=self.out_size, interm_channels=64)

    def forward(self, Y, M, M_fake):
        if len(Y) == 0:
            return None, None

        Y = Y.to(self.device)
        M = M.to(self.device)
        M_fake = M_fake.to(self.device)

        M_cat_Y = torch.cat((M, Y), dim=1)
        M_fake_cat_Y = torch.cat((M_fake, Y), dim=1)

        # discriminator score
        real_score = self.D(M_cat_Y)
        fake_score = self.D(M_fake_cat_Y)

        # cl loss
        loss = self.cl_loss_fn(real_score, fake_score, self.tau)
        if torch.isnan(loss):
            print("CL Loss NAN !!!")
            loss = torch.tensor(0)
        if loss <= 0:
            loss = torch.tensor(0)

        return loss

    def cl_loss_fn(self, p_samples, q_samples, tau):
        # Eq = torch.log(torch.sum(torch.exp(q_samples/tau), dim=0)).mean()
        # Eq = torch.mean(self.log_sum_exp(q_samples/tau, 0))
        Eq = torch.mean(torch.logsumexp(q_samples/tau, dim=0))

        # return - (p_samples/tau - Eq).mean()
        return np.log(len(q_samples)) - (p_samples/tau - Eq).mean()

    def log_sum_exp(self, x, axis=None):
        """Log sum exp function
        Args:
            x: Input.
            axis: Axis over which to perform sum.
        Returns:
            torch.Tensor: log sum exp
        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y

class Discriminator(torch.nn.Module):
    def __init__(self, M_channels, E_size, interm_channels=512):
        super().__init__()

        in_channels = E_size + M_channels
        self.c0 = torch.nn.Conv2d(in_channels, interm_channels, kernel_size=1)
        self.c1 = torch.nn.Conv2d(interm_channels, interm_channels, kernel_size=1)
        self.c2 = torch.nn.Conv2d(interm_channels, 1, kernel_size=1)

    def forward(self, x):               # torch.Size([32, 162, 1, 10])
        x = x.unsqueeze(-1).unsqueeze(-1)
        score = F.relu(self.c0(x))      # torch.Size([32, 64, 1, 10])
        score = F.relu(self.c1(score))  # torch.Size([32, 64, 1, 10])
        score = self.c2(score)          # torch.Size([32, 1, 1, 10])

        return score