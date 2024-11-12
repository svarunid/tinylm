# TinyLM

**TinyLM** is a framework designed to make training your own small language model simple and accessible. With TinyLM, you can train and fine-tune compact, efficient models on your own data, optimized for low-resource settings and local inference. Whether you're interested in building models for specific downstream tasks or exploring language model development, TinyLM offers a streamlined, user-friendly workflow.

## Key Features

- **Easy Configuration**: Define model settings, training parameters, and dataset paths in a `config.yaml` file.
- **Simple CLI Interface**: Train models with a single command: `tinylm train`.
- **Efficient for Local Inference**: Run models on personal devices or limited hardware with reduced memory and compute needs.
- **Customizable**: Supports flexible configuration options for model structure, training hyperparameters, and tokenization.

## Why TinyLM?

Recent trends favor smaller language models for several reasons:

- **Compute Efficiency**: Lower computational and memory requirements enable faster inference and lower costs.
- **Data Privacy**: Fine-tune models locally without exposing data to external servers.
- **Ease of Customization**: Small models can be easily adapted to specific tasks and domains.
- **Fast Experimentation**: Quick and efficient training iterations allow for rapid testing and experimentation.

## Installation

Clone the TinyLM repository:

```bash
git clone https://github.com/yourusername/tinylm.git
cd tinylm
```

Install dependencies:

```bash
pip install .
```

## Getting Started

### 1. Configure Model Settings

Edit the `config.yaml` file to define the model architecture, training settings, and dataset paths. Example:

```yaml
model:
  layers: 6
  hidden_size: 256
  vocab_size: 30000

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10

data: ...
```

### 2. Train the Model

Once configured, start training by running:

```bash
tinylm train
```

### 3. Fine-tune for Downstream Tasks

Use TinyLM’s configuration flexibility to fine-tune your model for specific tasks. Customize the training data path in `config.yaml` with your task-specific dataset and train the model.

## Future Directions

TinyLM is a work in progress with future plans to:

- Support model exporting for easy deployment.
- Provide more advanced hyperparameter tuning options.
- Enable training on distributed and multi-GPU systems.

## Contributing

Contributions are welcome! To get started, fork the repository, make your changes, and submit a pull request.
