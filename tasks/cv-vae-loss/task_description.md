# VAE Loss Function Design for Image Reconstruction

## Objective

Design a training loss function for a Variational Autoencoder (VAE) that achieves
the best reconstruction quality on CIFAR-10.

## Background

Variational Autoencoders encode images into a compressed latent representation and
decode them back. The quality of reconstruction depends critically on the training
loss function. Standard approaches use combinations of:

- **Reconstruction loss**: L1 or L2 pixel-level error
- **KL divergence**: Regularizes the latent space toward a standard normal prior
- **Perceptual loss**: LPIPS or VGG-based feature matching for perceptual quality
- **Adversarial loss**: Discriminator-based training for sharpness
- **Frequency-domain loss**: FFT-based weighting to preserve fine detail

Recent work on the Prism Hypothesis (UAE, Fan et al.) demonstrates that explicitly
handling different frequency bands in the training objective can significantly
improve reconstruction quality. The key insight is that semantic information
concentrates at low frequencies while fine perceptual detail lives in higher bands.

## Task

Implement the `VAELoss` class in `custom_train.py` (lines 32–76). Your loss
function will be used to train an `AutoencoderKL` model from the diffusers library
on CIFAR-10 32×32 images.

### Editable Region (lines 32–76)

```python
class VAELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Initialize your loss components here

    def forward(self, recon, target, posterior, step):
        # recon:     [B, 3, 32, 32] reconstructed images in [-1, 1]
        # target:    [B, 3, 32, 32] original images in [-1, 1]
        # posterior:  DiagonalGaussianDistribution
        #             - posterior.kl() -> KL divergence per sample
        #             - posterior.mean, posterior.logvar
        # step:       current training step (int)
        #
        # Return: (loss_tensor, metrics_dict)
        ...
```

### Available Libraries

- `torch`, `torch.nn`, `torch.nn.functional` — standard PyTorch
- `torch.fft` — frequency-domain operations (fft2, ifft2, fftshift, etc.)
- `lpips` — learned perceptual loss: `lpips.LPIPS(net='vgg').to(device)`
- `numpy`, `math`

### Architecture (Fixed)

The model is `AutoencoderKL` from diffusers with 3 blocks and 2 downsample
stages, giving latent resolution 8×8 (f=4 compression) suited for 32×32 input:
- `latent_channels=4`, `layers_per_block=2`
- GroupNorm (32 groups) + SiLU activation

Channel widths and latent channels scale via environment variables:
- Small: `BLOCK_OUT_CHANNELS=(64,128,256)`, `LATENT_CHANNELS=4` — lightweight
- Medium: `BLOCK_OUT_CHANNELS=(96,192,384)`, `LATENT_CHANNELS=8` — standard
- Large: `BLOCK_OUT_CHANNELS=(128,256,512)`, `LATENT_CHANNELS=16` — wide

### Training (Fixed)

- Optimizer: AdamW, lr=4e-4, weight_decay=1e-4
- LR schedule: 5% warmup + cosine decay
- Mixed precision (autocast + GradScaler)
- Gradient clipping at 1.0
- EMA with rate 0.999

## Evaluation

Reconstruction quality is measured on the full CIFAR-10 test set (10,000 images):

| Metric | Direction | Description |
|--------|-----------|-------------|
| **rFID** | lower is better | Reconstruction FID between original and reconstructed test images (primary metric) |
| **PSNR** | higher is better | Peak signal-to-noise ratio in dB |
| **SSIM** | higher is better | Structural similarity index |

Training scales:
- **Small** (group 1): 20,000 steps, channels (64,128,256), latent_channels=4
- **Medium** (group 2): 30,000 steps, channels (96,192,384), latent_channels=8
- **Large** (group 3): 30,000 steps, channels (128,256,512), latent_channels=16

## Baselines

| Name | Strategy | Description |
|------|----------|-------------|
| **l2-kl** | MSE + KL | Simplest VAE loss: pixel-level L2 reconstruction + KL regularization |
| **perceptual** | MSE + LPIPS + KL | Standard practice: adds learned perceptual similarity (VGG features) |
| **freq-weighted** | L1 + LPIPS + GAN + KL | Multi-objective: combines L1 reconstruction, LPIPS perceptual loss, and PatchGAN adversarial training |
