import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 2, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.beta = 0.2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.pre_quantization_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)
        self.post_quantization_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        pre_quantized_input = self.pre_quantization_conv(encoded)
        
        # Quantization
        B, C, H, W = pre_quantized_input.shape
        pre_quantized_input = pre_quantized_input.permute(0, 2, 3, 1)
        pre_quantized_input = pre_quantized_input.reshape((pre_quantized_input.size(0), -1, pre_quantized_input.size(-1)))
        
        # Calculating distances from embedding
        dist = torch.cdist(pre_quantized_input, self.embedding.weight[None, :].repeat((pre_quantized_input.size(0), 1, 1)))
        
        embedding_indices = torch.argmin(dist, dim=-1)
        
        # Select embeddings
        post_quantization_output = torch.index_select(self.embedding.weight, 0, embedding_indices.view(-1))
        
        pre_quantized_input = pre_quantized_input.reshape((-1, pre_quantized_input.size(-1)))
        
        # Calculate losses
        commitment_loss = torch.mean((post_quantization_output.detach() - pre_quantized_input) ** 2)
        codebook_loss = torch.mean((post_quantization_output - pre_quantized_input.detach()) ** 2)
        quantize_loss = codebook_loss + self.beta * commitment_loss
        
        post_quantization_output = pre_quantized_input * (post_quantization_output - pre_quantized_input).detach()
        post_quantization_output = post_quantization_output.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        embedding_indices = embedding_indices.reshape((-1, post_quantization_output.size(-2), post_quantization_output.size(-1)))
        
        # Passes through decoder
        decoder_input = self.post_quantization_conv(post_quantization_output)
        decoded = self.decoder(decoder_input)
        
        return decoded, quantize_loss
        