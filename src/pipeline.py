import click
import torch
from torchvision.transforms import ToTensor, ToPILImage, Lambda

from dataset.image_dataset import CosmosimImageDataset
from src.dataset.transformers import Transformer

def pipeline(training_data, test_data):
    return

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def main(root):
    root = "data/processed"
    training_data = CosmosimImageDataset(
        root=root,
        train=True,
        transform=Transformer.ALEXNET,
        target_transform=Lambda(lambda y: torch.tensor(y.astype(float).values))
    )

    test_data = CosmosimImageDataset(
        root=root,
        train=False,
        transform=Transformer.ALEXNET,
        target_transform=None
    )
    pipeline()

if __name__ == "__main__":
    main()