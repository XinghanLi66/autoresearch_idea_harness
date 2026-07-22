# Meta-Learning: Few-Shot Image Classification

## Objective
Design and implement a novel few-shot classification method. Your code goes in the `CustomFewShotMethod` class in `custom_fewshot.py`. Three reference implementations (Prototypical Networks, Matching Networks, Relation Networks) are provided as read-only.

## Background
Few-shot classification aims to recognize new classes from only a handful of labeled examples (the "support set"). During evaluation, the model receives a support set of N classes with K examples each (N-way K-shot), and must classify unlabeled query images into one of these N classes. Key design choices include:
- **Feature comparison**: how to measure similarity between query and support features (e.g., Euclidean distance, cosine similarity, learned metrics)
- **Support set encoding**: how to aggregate information from support examples (e.g., prototypes, attention, graph neural networks)
- **Query adaptation**: how to refine query representations using support context (e.g., LSTM attention, cross-attention, transductive inference)

## Model Interface
```python
class CustomFewShotMethod(FewShotClassifier):
    def __init__(self):
        # Create backbone and any learnable modules
        backbone = make_backbone(use_pooling=True)  # ResNet-12, output: [B, 640]
        super().__init__(backbone=backbone)

    def process_support_set(self, support_images, support_labels):
        # Extract and store support set information for later use
        ...

    def forward(self, query_images) -> Tensor:
        # Return classification scores of shape (n_query, n_way)
        ...

    def compute_loss(self, scores, labels) -> Tensor:
        # Compute training loss (default: cross-entropy)
        ...
```

## Available Utilities
- `self.compute_features(images)`: extract features through `self.backbone`
- `self.l2_distance_to_prototypes(features)`: negative Euclidean distance to `self.prototypes`
- `self.cosine_distance_to_prototypes(features)`: cosine similarity to `self.prototypes`
- `compute_prototypes(features, labels)`: compute mean feature per class
- `self.compute_prototypes_and_store_support_set(images, labels)`: convenience method
- `make_backbone(use_pooling=True/False)`: create ResNet-12 (640-dim features or feature maps)

## Evaluation
Trained and evaluated on three few-shot image classification benchmarks:
- **miniImageNet** (100 classes from ImageNet, 5-way 5-shot)
- **CIFAR-FS** (100 classes from CIFAR-100, 5-way 5-shot)
- **CUB-200** (200 fine-grained bird species, 5-way 5-shot)

Metric: mean classification accuracy over 600 test episodes (higher is better). Episodic training with 500 tasks/epoch for 200 epochs.
