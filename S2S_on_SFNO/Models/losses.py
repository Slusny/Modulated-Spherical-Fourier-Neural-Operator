import torch
import torch.nn as nn
import torch_harmonics as harmonics
import numpy as np

class CosineMSELoss(torch.nn.Module):
    def __init__(self, reduction='mean', eps=1e-4):
        super().__init__()
        self._mse = torch.nn.MSELoss(reduction='none')
        self.reduction = reduction  
        self.eps = eps

    def forward(self, x, y):
        B, C, H, W = x.shape

        weights = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=x.device, dtype=x.dtype))
        weights = torch.clamp(weights, min=0.0) 
        weights += self.eps 
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights[None, None, :, None]

        loss = self._mse(x, y)
        if self.reduction == "mean":
            return (loss * weights).mean() # dim=[2,3]
        if self.reduction == "sum":
            return (loss * weights).sum() / W
        if self.reduction == "none":
            return (loss * weights)  # B, C


# class CosineMSELoss(nn.Module):
#     def __init__(self, eps=1e-4, cache=True, with_poles=True,reduction=None):
#         super().__init__()
#         self.eps = eps
#         self.cache = cache
#         self.with_poles = with_poles

#         self._mse = nn.MSELoss(reduction='none')
#         self._weights = None

#     def extra_repr(self):
#         return "eps={eps}, cache={cache}, with_poles={with_poles}".format(**self.__dict__)

#     @torch.no_grad()
#     def _get_weights(self, latidudes):
#         if self.cache and self._weights is not None:
#             return self._weights
#         weights = torch.cos(latidudes * (2 * torch.pi / 360))
#         weights = torch.clamp(weights, min=0.0)  # ensure non-negative, this is very important due to numerical accuracy issues
#         weights += self.eps  # give poles a very small weight
#         weights /= weights.sum(dim=-1, keepdim=True)

#         if weights.dim() == 1:
#             weights = weights[None, None, :, None]  # Lat -> B,C,Lat,Lon
#         elif weights.dim() == 2:
#             weights = weights[:, None, :, None] # B, Lat -> B,C,Lat,Lon
        
#         if self.cache:
#             self._weights = weights

#         return weights


#     def forward(self, x, y, latitudes=None):
        
#         B, C, H, W = x.shape

#         if latitudes is None:
#             if self.with_poles:
#                 latitudes = torch.linspace(-90, 90, H, device=x.device, dtype=x.dtype)
#             else:
#                 latitudes = torch.linspace(-90 + 180 / H, 90 - 180 / H, H, device=x.device, dtype=x.dtype)
        
#         weights = self._get_weights(latitudes)
#         squared_diffs = self._mse(x, y)
#         loss = (squared_diffs * weights).sum(dim=[2,3]) / W

#         return loss  # B, C

class L2Sphere(torch.nn.Module):
    def __init__(self, relative=True, squared=False,reduction="sum",dampening=None):
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
        sphere_weights = torch.abs(sphere_weights)
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
            # mean is done by weights
            # return loss.mean()
            return loss.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class L2Sphere_noSine(torch.nn.Module):
    def __init__(self, relative=True, squared=False,reduction="sum",dampening=None):
        super(L2Sphere_noSine, self).__init__()
        
        self.relative = relative
        self.squared = squared
        self.reduction = reduction

    def forward(self, prd, tar):
        B, C, H, W = prd.shape
        w_quad = torch.tensor(harmonics.quadrature.legendre_gauss_weights(H, -1, 1)[1], device=prd.device, dtype=prd.dtype)
        # w_jacobian = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=prd.device, dtype=prd.dtype))
        # w_jacobian = w_jacobian[None, None, :, None]
        sphere_weights = w_quad 
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
            # mean is done by weights
            # return loss.mean()
            return loss.sum()
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

        # if reduction == 'mean':
        #     self.reduce = lambda x: x.mean()
        # elif reduction == 'sum':
        #     self.reduce = lambda x: x.sum()
        # elif reduction == 'none':
        #     self.reduce = lambda x: x
       
        if reduction in ['mean','sum','none']:
            self.reduction = reduction
        else:
            raise NotImplementedError(f'reduction method {reduction} not implemented')
       

    def forward(self,mu: torch.Tensor, sigma: torch.Tensor,  observation: torch.Tensor, mask: torch.Tensor = None):
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
        reduced_score = self.reduce(score,mask)
        return reduced_score

    def reduce(self,x,mask):
        if self.reduction == 'mean':
            if mask is not None:
                return x[mask].mean()
            else:
                return x.mean()
        elif self.reduction == 'sum':
            if mask is not None:
                return x[mask].sum()
            else:
                return x.sum() 
        elif self.reduction == 'none':
            # x[~mask] = torch.nan
            return x
    
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
        
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, observation: torch.Tensor):
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

    def forward(self, prediction: torch.Tensor, observation: torch.Tensor):
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