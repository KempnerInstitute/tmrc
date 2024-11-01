## What's Changed
TMRC now includes basic components to build and train transformer models.  We will continue to add features and optimize the code, but core functionality (models, optimizers, dataloading, etc.) are now present.  In particular this release:
* implements models through components, such as embeddings and layers, that are then assembled into architectures (such as GPT-type models)
* implements attention, document and causual masking through [FlexAttention](https://pytorch.org/blog/flexattention/)
* includes basic features to log and track your experiments on wandb
* includes basic documentation to get started on your platform