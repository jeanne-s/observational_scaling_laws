# Observational Scaling Laws and the Predictability of Language Model Performance

Re-implementation of the [Observational Scaling Laws](https://arxiv.org/abs/2405.10938) paper ([official repository](https://github.com/ryoungj/ObsScaling/tree/main)).

### Overview

The paper introduces an alternative method for predicting LLM performance without the need for training models across various scales. Traditional scaling laws rely on extensive compute resources to observe how performance changes with model size or training data, which is costly. The authors propose observational scaling laws, which leverage data from about 80 publicly available models to create scaling laws across different model families.

Their key finding is that performance can be generalized across model families through a low-dimensional capability space, which allows them to predict complex emergent behaviors, agentic capabilities, and the impact of post-training techniques, like Chain-of-Thought prompting. This method provides more cost-effective, higher-resolution scaling predictions and allows researchers to forecast the future behavior of models, such as GPT-4, using smaller models.

### Replication

```1_low-dim_capability_space.ipynb``` shows my replication of results presented in Part 3 of the paper.


