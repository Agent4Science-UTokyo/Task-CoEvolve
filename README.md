# Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection

Atsuyuki Miyai, Toshihiko Yamasaki, Kiyoharu Aizawa
The University of Tokyo

---

<p align="center">
  <img src="figs/overview.png" width="95%" alt="Overview of Task-CoEvolve">
</p>

## Abstract

We present a novel approach to efficient LLM agent harness optimization through
adaptive validation task selection. Harness optimization iteratively rewrites the
harness code based on validation performance, enabling substantial performance
gains without updating the underlying model weights. Existing approaches, however,
evaluate a fixed validation set in full at every iteration, incurring substantial
evaluation costs even on tasks that become less discriminative as the harness
evolves.

We propose **Task-CoEvolve**, which *co-evolves* the validation tasks with the
harness by addressing two challenges: selecting informative tasks and estimating
full-set performance from partial evaluations. Task-CoEvolve builds on the
observation that tasks on which candidate harnesses disagree are more informative
for distinguishing among them than tasks that are consistently solved or failed.
It uses variance-weighted sampling based on past outcomes to focus evaluation on
tasks near the agent's capability frontier, with the sampling distribution
adapting as the harness evolves. It then estimates full-set scores from the
sampled tasks by accounting for their sampling probabilities, enabling consistent
comparisons across iterations despite evaluating different subsets.

Experiments on online text classification and Terminal-Bench 2.1 show that
Task-CoEvolve consistently outperforms fixed-subset baselines and matches the
final performance of full-set search while reducing the number of evaluations
during optimization by 80%.

## Code

**Code will be available soon.**

## Citation

```bibtex
@article{miyai2026taskcoevolve,
  title   = {Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection},
  author  = {Miyai, Atsuyuki and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
  journal = {arXiv preprint},
  year    = {2026}
}
```
