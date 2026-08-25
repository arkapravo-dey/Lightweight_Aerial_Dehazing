import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="GRSL Aerial Image Dehazing"
    )

    # Output directories
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")

    # SateHaze1K dataset
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--val_thick_dir", type=str, default="")
    parser.add_argument("--val_moderate_dir", type=str, default="")
    parser.add_argument("--val_thin_dir", type=str, default="")
    parser.add_argument("--test_thick_dir", type=str, default="")
    parser.add_argument("--test_moderate_dir", type=str, default="")
    parser.add_argument("--test_thin_dir", type=str, default="")

    # RICE1 dataset
    parser.add_argument("--rice_train_dir", type=str, default="")
    parser.add_argument("--rice_val_dir", type=str, default="")

    # Model
    parser.add_argument("--model", type=str, default="GRSL_AERIAL_DEHAZING_MODEL")
    parser.add_argument("--load_model_path", type=str, default="")

    # Training
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--eta_min", type=float, default=1e-7)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="ADAM",
        choices=["ADAM", "ADAMW"],
    )

    return parser.parse_args()