import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask, scale_fn):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask = mask
        self.scale_fn = scale_fn

        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

        self.log_scale_factor = nn.Parameter(torch.zeros(input_dim // 2))

    def scale_fn_example(x):
        return torch.tanh(x) * 0.5 + 0.5
        
    def forward(self, x):
        x1, x2 = self.mask * x, (1 - self.mask) * x  #Chooses which features to split x1 and x2
        
        s = self.scale_fn(self.net(x1))
        z2 = x2 * torch.exp(s) + self.net(x1) #Correct
        
        log_det = torch.sum(s, dim=1) + self.log_scale_factor
        return torch.cat([x1, z2], dim=1), log_det

    def inv(self, z):
        z1, z2 = self.mask * z, (1 - self.mask) * z
        s = self.scale_fn(self.net(z1))
        x1 = z1
        x2 = (z2 - self.net(z1)) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)
        log_det = -torch.sum(s, dim=1) - self.log_scale_factor
        return x, log_det
    
    
    
    
    
    
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RealNVP, self).__init__()

        # Initialize the list of affine coupling layers
        self.transforms = nn.ModuleList([])
        for i in range(num_layers):
            self.transforms.append(AffineCoupling(input_dim, hidden_dim))

        # Set the base distribution to a standard normal distribution
        self.base_dist = torch.distributions.Normal(0, 1)

        # Save the input dimension and number of layers
        self.input_dim = input_dim
        self.num_layers = num_layers

    def split(self, x):
        # Generate a random permutation of the indices along the dim=1 dimension
        perm = torch.randperm(x.shape[1])

        # Use the first half of the permutation to split x into two subsets
        split_idx = x.shape[1] // 2
        x1, x2 = x[:, perm[:split_idx]], x[:, perm[split_idx:]]

        return x1, x2

    def merge(self, x1, x2):
        # Concatenate x1 and x2 along the dim=1 dimension
        x = torch.cat((x1, x2), dim=1)

        return x

    def forward(self, x):
        # Split the input tensor x into two subsets x1 and x2
        x1, x2 = self.split(x)

        # Transform x1 using the affine coupling layers
        z1 = x1
        for i in range(self.num_layers):
            z1, _ = self.transforms[i](z1)

        # Merge z1 and x2 to obtain the output tensor
        z = self.merge(z1, x2)
        
        return z

    def inverse(self, z):
        # Split the input tensor z into two subsets z1 and x2
        z1, x2 = self.split(z)

        # Transform z1 using the inverse affine coupling layers
        x1 = z1
        for i in reversed(range(self.num_layers)):
            x1, _ = self.transforms[i].inv(x1)

        # Merge x1 and x2 to obtain the input tensor
        x = self.merge(x1, x2)

        return x

    def log_prob(self, x):
        # Transform x using the inverse transformations in reverse order
        z = x
        for i in reversed(range(self.num_layers)):
            z = self.transforms[i].inv(z)

        # Compute the log-likelihood of z under the base distribution
        log_prob = self.base_dist.log_prob(z)

        # Compute the log-determinant of the Jacobian matrix of the transformations
        log_det = torch.cat([t.log_det() for t in self.transforms]).sum(dim=0)

        # Compute the log-likelihood of x under the flow model
        log_likelihood = log_prob + log_det

        return log_likelihood

    def loss_function(self, x):
        return -self.log_prob(x).mean()