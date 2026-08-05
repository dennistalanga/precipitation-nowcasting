def get_model(
    model_name,
    sequence_length,
    predict_steps,
    input_channels=2,
    base_channels=32,
    num_groups=4
):
    match model_name:
        case "baseline_cnn":
            from src.models.baseline_cnn import BaselineCNN
            return BaselineCNN(
                sequence_length=sequence_length,
                predict_steps=predict_steps,
                input_channels=input_channels,
                base_channels=base_channels,
                num_groups=num_groups
            )
            
        case "conv_lstm":
            from src.models.conv_lstm import ConvLSTM
            return ConvLSTM()
            
        case "unet":
            from src.models.unet import UNet
            return UNet()
            
        case "transformer":
            from src.models.transformer import Transformer
            return Transformer()
            
        case _:
            raise ValueError(f"Unknown model: {model_name}")
