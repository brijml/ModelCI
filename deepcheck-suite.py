import sys

import numpy as np
import torch
from deepchecks.vision import Suite, VisionData
from deepchecks.vision.checks.model_evaluation import *
from deepchecks.vision.utils.image_properties import brightness, texture_level
from deepchecks.vision.vision_data import BatchOutputFormat
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from train import NeuralNetwork


def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    """Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with
    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.
    You can also use the BatchOutputFormat class to create the output.
    """
    # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
    batch = tuple(zip(*batch))

    # images:
    inp = torch.stack(batch[0]).detach().numpy().transpose((0, 2, 3, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    images = np.clip(inp, 0, 1) * 255

    # labels:
    labels = batch[1]

    # predictions:
    logits = model.to(device)(torch.stack(batch[0]).to(device))
    predictions = nn.Softmax(dim=1)(logits)
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)


def get_class_performance_check(threshold):
    check = ClassPerformance()
    check.add_condition_test_performance_greater_than(threshold)
    return check


def get_simple_model_comparison_check(strategy, threshold):
    check = SimpleModelComparison(strategy=strategy)
    check.add_condition_gain_greater_than(min_allowed_gain=threshold)
    return check


def get_weak_segments_performance_check(threshold):
    properties = [
        {"name": "brightness", "method": brightness, "output_type": "numerical"},
        {"name": " texture", "method": texture_level, "output_type": "numerical"},
    ]
    check = WeakSegmentsPerformance(
        segment_minimum_size_ratio=threshold, image_properties=properties
    )
    return check


if __name__ == "__main__":

    LABEL_MAP = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    model = torch.load("model.pt")
    device = "cpu"

    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn
    )

    training_data = VisionData(
        batch_loader=train_loader, task_type="classification", label_map=LABEL_MAP
    )
    test_data = VisionData(
        batch_loader=test_loader, task_type="classification", label_map=LABEL_MAP
    )

    # Initialize the Test suite
    suite = Suite(
        "Custom Suite for testing classification model",
        get_simple_model_comparison_check(strategy="stratified", threshold=0.99),
        get_class_performance_check(0.2),
        get_weak_segments_performance_check(0.33),
    )
    result = suite.run(train_dataset=training_data, test_dataset=test_data)
    result.save_as_html("output.html")

    if not result.passed():
        sys.exit(1)
