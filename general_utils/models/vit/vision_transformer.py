from timm.models.vision_transformer import VisionTransformer

def vit_tiny_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=192,
                             depth=12,
                             num_heads=3,
                             **kwargs)


def vit_small_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=384,
                             depth=12,
                             num_heads=6,
                             **kwargs)


def vit_medium_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=512,
                             depth=12,
                             num_heads=8,
                             **kwargs)


def vit_base_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=768,
                             depth=12,
                             num_heads=12,
                             **kwargs)

def vit_large_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1024,
                             depth=24,
                             num_heads=16,
                             **kwargs)


def vit_huge_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1280,
                             depth=32,
                             num_heads=16,
                             **kwargs)


def vit_giant_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1408,
                             depth=40,
                             num_heads=16,
                             **kwargs)


def vit_gigantic_patch4_32(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=32,
                             patch_size=4,
                             embed_dim=1664,
                             depth=48,
                             num_heads=16,
                             **kwargs)


def vit_tiny_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=192,
                             depth=12,
                             num_heads=3,
                             **kwargs)


def vit_small_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=384,
                             depth=12,
                             num_heads=6,
                             **kwargs)


def vit_medium_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=512,
                             depth=12,
                             num_heads=8,
                             **kwargs)


def vit_base_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=768,
                             depth=12,
                             num_heads=12,
                             **kwargs)

def vit_large_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1024,
                             depth=24,
                             num_heads=16,
                             **kwargs)


def vit_huge_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1280,
                             depth=32,
                             num_heads=16,
                             **kwargs)


def vit_giant_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1408,
                             depth=40,
                             num_heads=16,
                             **kwargs)


def vit_gigantic_patch16_224(n_classes: int, n_channels: int, **kwargs):
    return VisionTransformer(num_classes=n_classes,
                             in_chans=n_channels,
                             img_size=224,
                             patch_size=16,
                             embed_dim=1664,
                             depth=48,
                             num_heads=16,
                             **kwargs)

