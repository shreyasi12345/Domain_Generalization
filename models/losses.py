import torch
import torch.nn.functional as F


class EntropyMaximization(torch.nn.Module):
    """Entropy Maximization loss

    Arguments:
        t : temperature
    """

    def __init__(self, t=1.):
        super(EntropyMaximization, self).__init__()
        self.t = t

    def forward(self, lbl, pred):
        """Compute loss.

        Arguments:
            lbl (torch.tensor:float): predictions, not confidence, not label.
            pred (torch.tensor:float): predictions.

        Returns:
            loss (torch.tensor:float): entropy maximization loss

        """
        loss = torch.mean(torch.sum(F.softmax(lbl/self.t, dim=-1) * F.log_softmax(pred/self.t, dim=-1), dim=-1))
        return loss


class MMDLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        """
        Initialize the MMD Loss module.

        Arguments:
            sigma (float): Bandwidth parameter for the RBF kernel.
        """
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def rbf_kernel(self, x, y):
        """
        Compute the RBF kernel between two tensors.

        Arguments:
            x (torch.tensor:float): Source tensor of shape (n_samples, n_features).
            y (torch.tensor:float): Target tensor of shape (m_samples, n_features).

        Returns:
            torch.tensor: Pairwise RBF kernel values of shape (n_samples, m_samples).
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # Pairwise differences
        dist_sq = torch.sum(diff ** 2, dim=-1)  # Squared distances
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))

    def forward(self, lbl, pred):
        """
        Compute MMD Loss.

        Arguments:
            lbl (torch.tensor:float): Source domain features of shape (n_samples, n_features).
            pred (torch.tensor:float): Target domain features of shape (m_samples, n_features).

        Returns:
            loss (torch.tensor:float): MMD loss value.
        """
        # Compute kernels
        k_lbl_lbl = self.rbf_kernel(lbl, lbl)  # Source-to-Source
        k_pred_pred = self.rbf_kernel(pred, pred)  # Target-to-Target
        k_lbl_pred = self.rbf_kernel(lbl, pred)  # Source-to-Target

        # Compute MMD loss
        loss = k_lbl_lbl.mean() + k_pred_pred.mean() - 2 * k_lbl_pred.mean()
        return loss




class DomainAdversarialLoss(torch.nn.Module):
    def __init__(self, t=1.0):
        """
        Initialize the Domain Adversarial Loss.

        Arguments:
            t (float): Temperature scaling factor.
        """
        super(DomainAdversarialLoss, self).__init__()
        self.t = t  # Temperature scaling factor

    def forward(self, lbl, pred):
        """
        Compute domain adversarial loss using class predictions.

        Arguments:
            lbl (torch.tensor:float): Class predictions from the source domain.
            pred (torch.tensor:float): Class predictions from the target domain.

        Returns:
            loss (torch.tensor:float): Domain adversarial loss.
        """
        # Compute softmax probabilities
        source_probs = F.softmax(lbl / self.t, dim=-1)
        target_probs = F.softmax(pred / self.t, dim=-1)

        # Align distributions using Jensen-Shannon divergence
        # Compute the mean of the distributions
        mean_probs = 0.5 * (source_probs + target_probs)

        # Jensen-Shannon divergence = 0.5 * KL(source || mean) + 0.5 * KL(target || mean)
        kl_source = torch.sum(source_probs * torch.log(source_probs / mean_probs + 1e-8), dim=-1)
        kl_target = torch.sum(target_probs * torch.log(target_probs / mean_probs + 1e-8), dim=-1)

        js_divergence = 0.5 * (kl_source + kl_target)

        # Loss is the mean of the JS divergence across the batch
        loss = torch.mean(js_divergence)

        return loss

