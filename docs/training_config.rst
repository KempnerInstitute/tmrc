Training Configuration
======================

The training configuration is managed through a YAML file with several main sections: ``datasets``, ``model``, ``optimizer``, ``tokenizer``, ``training``, and logging settings.

Dataset Configuration
---------------------
Configuration related to input data and tokenization.

.. code-block:: yaml

    datasets:
      name: algebraic-stack
      path: "/path/to/data"
      tokenizer_used: t5-base

* ``name``: Name of the dataset being used (that was tokenized by the ``tatm`` package)
* ``path``: File system path to the tokenized dataset
* ``tokenizer_used``: The tokenizer that was used to preprocess the data

Model Configuration
-------------------
Parameters that define the model architecture.

.. code-block:: yaml

    model:
      name: gpt
      n_head: 4
      d_model: 512
      n_layer: 8
      dropout_p: 0.0
      context_length: 512
      autocast_precision: bfloat16
      flash: False
      flex: True
      mlp_scale_factor: 4
      mlp_bias: True
      attn_bias: False
      proj_bias: True
      ln_bias: True
      cls_head_bias: True
      activation: relu
      mask: causal_document

* ``name``: Model architecture type (currently supports 'gpt')
* ``n_head``: Number of attention heads
* ``d_model``: Hidden dimension size
* ``n_layer``: Number of transformer layers
* ``dropout_p``: Dropout probability (0.0 means no dropout)
* ``context_length``: Maximum sequence length for input tokens
* ``autocast_precision``: Precision for automatic mixed precision training (options: ``float32``, ``float16``, ``bfloat16``)
* ``flash``: Whether to use Flash Attention for faster computation
* ``flex``: Whether to use Flex Attention (cannot be used with Flash Attention)
* ``mlp_scale_factor``: Multiplier for MLP hidden dimension relative to ``d_model``
* ``mlp_bias``: Include bias terms in MLP layers
* ``attn_bias``: Include bias terms in attention computation
* ``proj_bias``: Include bias terms in projection layers
* ``ln_bias``: Include bias terms in layer normalization
* ``cls_head_bias``: Include bias terms in classification head
* ``activation``: Activation function (options: ``relu``, ``gelu``)
* ``mask``: Attention mask type (e.g., ``causal_document`` to use causal attention + document masking).  If document masking is used, this requires Flex Attention to be enabled.

Optimizer Configuration
-----------------------
Parameters for the optimization algorithm.

.. code-block:: yaml

    optimizer:
      name: AdamW
      lr: 0.0001
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8
      precision: float32

* ``name``: Optimizer type (currently supports 'AdamW')
* ``lr``: Learning rate
* ``weight_decay``: L2 regularization factor
* ``betas``: Beta parameters for AdamW [β1, β2]
* ``eps``: Epsilon parameter for numerical stability
* ``precision``: Optimizer state precision

Tokenizer Configuration
-----------------------
Settings for the tokenizer.

.. code-block:: yaml

    tokenizer:
      name: t5-base
      vocab_size: 32128

* ``name``: Name of the pretrained tokenizer
* ``vocab_size``: Size of the vocabulary

Training Configuration
----------------------
Parameters controlling the training process.

.. code-block:: yaml

    training:
      epochs: 1
      train_steps: 100000
      batch_size: 256
      log_interval: 20
      shuffle: True
      save_model: True
      save_every: 3600
      artifacts_path: /path/to/artifacts
      use_oracle: False

* ``epochs``: Number of training epochs
* ``train_steps``: Maximum number of training steps (training stops at whichever comes first: epochs or train_steps)
* ``batch_size``: Size of training batches
* ``log_interval``: Number of steps between logging updates
* ``shuffle``: Whether to shuffle the dataset between epochs
* ``save_model``: Whether to save model checkpoints
* ``save_every``: Checkpoint saving frequency in seconds (3600 = once an hour)
* ``artifacts_path``: Directory to save model checkpoints and other artifacts
* ``use_oracle``: Enable oracle mode for debugging/testing

Logging Configuration
---------------------
Settings for experiment tracking.

.. code-block:: yaml

    wandb_log:
      name: tmrc_log

* ``name``: Run name for Weights & Biases logging

Hydra Configuration
-------------------
Settings for Hydra configuration management.

.. code-block:: yaml

    HydraConf:
      version_base: "1.1"

* ``version_base``: Hydra version compatibility setting