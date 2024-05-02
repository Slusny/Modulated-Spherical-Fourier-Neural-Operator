import torch
import torch.nn as nn
import torch_harmonics as harmonics

class CosineMSELoss():
    def __init__(self, reduction=None):
        super().__init__()
        self._mse = torch.nn.MSELoss(reduction='none')
        self.reduction = reduction  

    def forward(self, x, y):
        B, C, H, W = x.shape

        weights = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=x.device, dtype=x.dtype))
        weights /= weights.sum()
        weights = weights[None, None, :, None]

        loss = self._mse(x, y)
        loss = (loss * weights).sum(dim=[2,3]) / W
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss  # B, C

class L2Sphere(torch.nn.Module):
    def __init__(self, relative=True, squared=False,reduction="sum"):
        super(L2Sphere, self).__init__()
        
        self.relative = relative
        self.squared = squared
        self.reduction = reduction

    def forward(self, prd, tar):
        B, C, H, W = prd.shape
        w_quad = torch.tensor(harmonics.quadrature.legendre_gauss_weights(H, -1, 1)[1], device=prd.device, dtype=prd.dtype)
        w_jacobian = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=prd.device, dtype=prd.dtype))
        # w_jacobian = w_jacobian[None, None, :, None]
        sphere_weights = w_quad * w_jacobian
        sphere_weights = sphere_weights[None, None, :, None]   

        if self.reduction == "none":
            loss = (sphere_weights*(prd - tar)**2)
            if self.relative:
                loss = loss / (sphere_weights*tar**2).sum(dim=(-1,-2))
            return loss
        
        loss = (sphere_weights*(prd - tar)**2).sum(dim=(-1,-2))
        if self.relative:
            loss = loss / (sphere_weights*tar**2).sum(dim=(-1,-2))
        
        if not self.squared:
            loss = torch.sqrt(loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def spectral_loss_sphere(solver, prd, tar, relative=False, squared=True):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls*(ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    coeffs = spectral_weights * coeffs
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_coeffs = spectral_weights * tar_coeffs
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def h1loss_sphere(solver, prd, tar, relative=False, squared=True):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls*(ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    h1_coeffs = spectral_weights * coeffs
    h1_norm2 = h1_coeffs[..., :, 0] + 2 * torch.sum(h1_coeffs[..., :, 1:], dim=-1)
    l2_norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    h1_loss = torch.sum(h1_norm2, dim=(-1,-2))
    l2_loss = torch.sum(l2_norm2, dim=(-1,-2))

     # strictly speaking this is not exactly h1 loss 
    if not squared:
        loss = torch.sqrt(h1_loss) + torch.sqrt(l2_loss)
    else:
        loss = h1_loss + l2_loss

    if relative:
        raise NotImplementedError("Relative H1 loss not implemented")

    loss = loss.mean()


    return loss

def fluct_l2loss_sphere(solver, prd, tar, inp, relative=False, polar_opt=0):
    # compute the weighting factor first
    fluct = solver.integrate_grid((tar - inp)**2, dimensionless=True, polar_opt=polar_opt)
    weight = fluct / torch.sum(fluct, dim=-1, keepdim=True)
    # weight = weight.reshape(*weight.shape, 1, 1)
    
    loss = weight * solver.integrate_grid((prd - tar)**2, dimensionless=True, polar_opt=polar_opt)
    if relative:
        loss = loss / (weight * solver.integrate_grid(tar**2, dimensionless=True, polar_opt=polar_opt))
    loss = torch.mean(loss)
    return loss



class NormalCRPS(nn.Module):
    '''
    Continuous Ranked Probability Score (CRPS) loss for a normal distribution as described in the paper "Probabilistic Forecasting with Gated Neural Networks"
    '''
    def __init__(self, reduction = 'mean', sigma_transform = 'softplus'):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the std estimate, can be 'softplus', 'exp' or 'none'
        '''
        super().__init__()
        self.sqrtPi = torch.as_tensor(np.pi).sqrt()
        self.sqrtTwo = torch.as_tensor(2.).sqrt()

        if sigma_transform == 'softplus':
            self.sigma_transform = lambda x: nn.functional.softplus(x)
        elif sigma_transform == 'exp':
            self.sigma_transform = lambda x: torch.exp(x)
        elif sigma_transform == 'none':
            self.sigma_transform = lambda x: x
        else:
            raise NotImplementedError(f'Sigma transform {sigma_transform} not implemented')

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')

    def forward(self, observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        '''
        Compute the CRPS for a normal distribution
            :param observation: (batch, *) tensor of observations
            :param mu: (batch, *) tensor of means
            :param log_sigma: (batch, *) tensor of log standard deviations
            :return: CRPS score     
            '''
        std = self.sigma_transform(sigma) #ensure positivity
        z = (observation - mu) / std #z transform
        phi = torch.exp(-z ** 2 / 2).div(self.sqrtTwo * self.sqrtPi) #standard normal pdf
        score = std * (z * torch.erf(z / self.sqrtTwo) + 2 * phi - 1 / self.sqrtPi) #crps as per Gneiting et al 2005
        reduced_score = self.reduce(score)
        return reduced_score
    
class Beta_NLL(nn.Module):
    '''
    Beta Negative Log Likelihood loss as described in the paper "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
    '''
    def __init__(self, beta: float = 0.5, reduction: str = 'mean', sigma_transform: str = 'softplus') -> None:
        super().__init__()
        '''
        beta: the beta parameter that controls the tradeoff between the mean and the variance of the predictive distribution
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the variance estimate, can be 'softplus', 'exp' or 'none'
        '''
        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')
        
        if sigma_transform == 'softplus':
            self.sigma_transform = lambda x: nn.functional.softplus(x)
        elif sigma_transform == 'exp':
            self.sigma_transform = lambda x: torch.exp(x)
        elif sigma_transform == 'none':
            self.sigma_transform = lambda x: x
        else:
            raise NotImplementedError(f'Sigma transform {sigma_transform} not implemented')
        
        self.beta = beta
        
    def forward(self, observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        '''
        Calculates the beta nll as described in the paper "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
        :param observation: the observation
        :param mu: the mean of the predictive distribution
        :param variance: the variance of the predictive distribution
        '''
        variance = self.sigma_transform(sigma)
        loss = 0.5 * (((observation - mu) ** 2) / variance + torch.log(variance))
        if self.beta > 0:
            loss = loss * (variance.detach() ** self.beta)
        return self.reduce(loss)
    
class StatisticalLoss(nn.Module):
    '''
    Statistical loss function as defined in 'AtmoRep: A stochastic model of atmosphere dynamics using large scale representation learning'
    '''
    def __init__(self, reduction = 'mean', ensemble_dim = -1):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        '''
        super().__init__()

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')
        
        self.ensemble_dim = ensemble_dim

    def forward(self, observation: torch.Tensor, prediction: torch.Tensor):
        '''
        Compute the first order statistical loss from ensemble predictions
            :param observation: (batch, *) tensor of observations
            :param prediction: (batch, *, ensemble) tensor of ensemble predictions
            :return: CRPS score     
            '''
        #calculate first order ensemble statistics
        mu = prediction.mean(dim = self.ensemble_dim)
        sigma = prediction.std(dim = self.ensemble_dim)
        #calculate unnormalized Gaussian likelihood
        phi = torch.exp(((mu - observation) / sigma).pow(2).div(2))
        #calculate squared distance between the Gaussian and the Dirac likelihood
        stat_dist = (1 - phi).pow(2)
        #calculate squared distance between each ensemble member and the observation
        member_dist = (prediction - observation.unsqueeze(-1)).pow(2).sum(-1)
        #regularization term controling the variance
        var_regularization = sigma.sqrt()
        #total score is the sum of the three terms
        score = stat_dist + member_dist + var_regularization
        #apply reduction
        reduced_score = self.reduce(score)
        return reduced_score