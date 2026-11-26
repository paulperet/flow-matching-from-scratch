import argparse
from pathlib import Path

from utils.training_loop import train

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model to train: ae, vae or b-vae (default: ae)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default='output.pth',
        help="Name of the model",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=False,
        help="learning rate",
        default=1e-4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        help="num workers",
        default=0
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_file=args.output_file,
        num_workers=args.num_workers,
    )