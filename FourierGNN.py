import torch
import torch.nn as nn
import torch.nn.functional as F

class FGN(nn.Module):
    def __init__(self, args, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.adj_matrix1 = nn.Parameter(torch.randn(1, 1))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.args = args
        self.to(args.device)

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, edge_index, N, L):
        o1_real = torch.zeros([1, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([1, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)
        if self.adj_matrix1.shape != ((N*L)//2 + 1, (N*L)//2 + 1):  # 重新初始化
            self.adj_matrix1 = nn.Parameter(torch.randn((N*L)//2 + 1, (N*L)//2 + 1).to(x.device))

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_real = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o1_real)

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        o1_imag = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o1_imag)

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_real = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o2_real)

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        o2_imag = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o2_imag)

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_real = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o3_real)

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        o3_imag = torch.einsum("ij, bjk -> bik", self.adj_matrix1, o3_imag)

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x, edge_index):
        N, L = x.shape
        x = x.reshape(-1, N*L)
        # embedding NL ==> NL*D
        x = self.tokenEmb(x)
        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        x = x.reshape(1, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, edge_index, N, L)

        x = x + bias

        x = x.reshape(1, (N*L)//2+1, self.embed_size)
        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        x = x.reshape(N, self.embed_size, L)
        # x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(N, -1)
        x = self.fc(x)
        return x
