#!/usr/bin/env python3
"""Basic tests for CI/CD pipeline."""

import yaml
import torch
from pathlib import Path
from test import load_config
from models.unet_fft import UNet_FFT
from models.xrd_transformer import XRDTransformer

def test_config_loading():
    """Test if configuration can be loaded properly."""
    config = load_config('configs/test.yaml')
    assert isinstance(config, dict), 'Config should be a dictionary'
    
    # Test essential config keys
    required_keys = ['data_path', 'model', 'pipeline', 'dataloader']
    for key in required_keys:
        assert key in config, f'Config missing required key: {key}'
    
    return "Тест загрузки конфигурации прошел успешно ✅"

def test_model_initialization():
    """Test if models can be initialized properly."""
    # Test UNet_FFT
    unet = UNet_FFT(2)
    assert isinstance(unet, torch.nn.Module), 'UNet_FFT should be a torch.nn.Module'
    
    # Test XRDTransformer
    transformer = XRDTransformer(
        input_shape=(26, 18, 23),
        embed_dim=128,
        depth=5,
        num_heads=4,
        mlp_ratio=4,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        embedding_type='onehot'
    )
    assert isinstance(transformer, torch.nn.Module), 'XRDTransformer should be a torch.nn.Module'
    
    return "Тест инициализации модели прошел успешно ✅"

def test_model_forward():
    """Test if models can process data of expected shape."""
    model = UNet_FFT(2)
    dummy_input = torch.randn(1, 1, 26, 18, 23)
    output = model(dummy_input)
    
    assert output.shape == dummy_input.shape, 'Output shape should match input shape'
    assert not torch.isnan(output).any(), 'Output contains NaN values'
    assert not torch.isinf(output).any(), 'Output contains infinite values'
    
    return "Тест проверки прямого прохода модели прошел успешно ✅"

def run_all_tests():
    """Run all tests and return results."""
    try:
        results = []
        results.append(test_config_loading())
        results.append(test_model_initialization())
        results.append(test_model_forward())
        
        return "\n".join(results)
    except Exception as e:
        return f"❌ Тесты не прошли: {str(e)}"

if __name__ == "__main__":
    print(run_all_tests())