
+++
title = "More on Adam"
draft = false
[build] 
list = "never"
render = "always"
math = true
+++

# Adam and AdamW Optimizers

Optimization algorithms are essential for training deep learning models. They determine how model parameters (weights and biases) are updated during backpropagation to minimize the loss function.

Among the most widely used optimizers in modern deep learning are **Adam** and its improved variant **AdamW**. Both are based on adaptive learning rate methods that adjust how much each parameter changes based on historical gradients.

---

## The Adam Optimizer

**Adam** (short for *Adaptive Moment Estimation*) was introduced by Diederik P. Kingma and Jimmy Ba in 2014.  
It combines the strengths of two earlier methods; **AdaGrad** (adaptive learning rates for sparse data) and **RMSProp** (exponential moving average of squared gradients).  
  [See More](https://arxiv.org/abs/1412.6980)

### Core Concepts

Adam maintains two moving averages for each parameter during training:

1. **First moment (mean of gradients):**  
   Tracks the average direction of updates.

2. **Second moment (uncentered variance):**  
   Tracks the magnitude of gradients to scale learning adaptively.

These are computed as:


$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$  
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 
$$


- g sub t is the gradient at time step (t)


- beta 1 and beta 2 are decay rates (typically 0.9 and 0.999)
- m sub t and v sub t are bias-corrected before being applied.

The parameter update rule is then:

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
$$

where:

- alpha is the learning rate  
- epsilon is a small constant for numerical stability  

### Advantages

- **Adaptive learning rates:** Each parameter adjusts based on its own gradient history.  
- **Fast convergence:** Works well for large-scale, noisy datasets.  
- **Minimal tuning:** Performs reliably with default parameters.

### Limitations

Despite its effectiveness, the original Adam algorithm can sometimes **generalize poorly** compared to SGD.  
This happens because weight decay (regularization) is coupled with adaptive learning rates, leading to inconsistent regularization strength, a problem that **AdamW** addresses.

---

## The AdamW Optimizer

**AdamW** (short for *Adam with Decoupled Weight Decay*) was introduced by Ilya Loshchilov and Frank Hutter in 2017.  
It modifies Adam by **decoupling weight decay from the gradient-based update rule**, improving both training stability and generalization.  
  [See More](https://arxiv.org/abs/1711.05101)

### Core Idea

In standard Adam, weight decay was implemented by adding the L2 regularization term directly into the gradient computation. This unintentionally made weight decay **dependent on the adaptive learning rate**, distorting the intended regularization effect.

AdamW fixes this by applying weight decay as a **separate step** after gradient updates:

$$
\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} + \lambda \theta_{t-1} \right)
$$

where:

- lambda is the weight decay coefficient  
- alpha is the learning rate  

This separation ensures consistent regularization strength regardless of learning rate adjustments.

### Benefits of AdamW

- **Improved generalization:** Matches or exceeds SGD performance on large-scale tasks (e.g., ImageNet, NLP fine-tuning).  
- **Stable training:** Decoupling weight decay prevents unintended interference with adaptive gradients.  
- **Standardization:** Adopted as the default optimizer for Transformer-based architectures (e.g., BERT, RoBERTa, GPT).  

[← Back to previous page](/projects/yt-classifier/)