# Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection

[Atsuyuki Miyai](https://atsumiyai.github.io/), [Kiyoharu Aizawa](https://sites.google.com/view/aizawa-kiyoharu)\*, [Toshihiko Yamasaki](https://scholar.google.com/citations?user=rE9iY5MAAAAJ&hl=ja)\*  
The University of Tokyo  
\*Equal supervision

📄 **Paper:** [alphaXiv 2608.20169](https://www.alphaxiv.org/ja/abs/2608.20169) &nbsp;|&nbsp; 🎬 **Video:** [YouTube](https://www.youtube.com/watch?v=MvI17TNFW9c&t=61s)

---

<p align="center">
  <img src="figs/overview.png" width="95%" alt="Overview of Task-CoEvolve">
</p>

## TL;DR

> **Same performance as full-set harness optimization, with 80% fewer evaluations.**
>
> Task-CoEvolve co-evolves the validation task set with the harness: it focuses evaluation on tasks where candidate harnesses disagree, and corrects the resulting estimates so scores stay comparable across iterations. On online text classification and Terminal-Bench 2.1, it **matches the final performance of full-set search while cutting the evaluation budget during optimization by 80%**, and consistently outperforms subset-based baselines at the same budget.

## Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=MvI17TNFW9c&t=61s">
    <img src="figs/youtube.png" width="95%" alt="Task-CoEvolve video walkthrough on YouTube">
  </a>
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

Task-CoEvolve is implemented in `extensions/` inside each experimental setting:

| Directory | Setting |
| --- | --- |
| [`reference_examples/text_classification/`](reference_examples/text_classification/) | Online text classification (memory-system search) |
| [`reference_examples/terminal_bench_2/`](reference_examples/terminal_bench_2/) | Terminal-Bench 2.1 (agent scaffold search) |

### Setup

Each setting is its own uv project with its own requirements; see its README for
the full setup (install, proposer login, model credentials, smoke check). In
short:

1. Install [uv](https://docs.astral.sh/uv/) and Node.js / npm. Terminal-Bench
   2.1 also needs Docker.
2. Copy the credentials template and fill in the keys for the models you run:

   ```bash
   cp .env.example .env
   ```

3. In the setting's directory:

   ```bash
   uv sync
   bash scripts/install_claude_cli.sh    # pins the proposer CLI version
   ```

4. Log in to the proposer CLI with a Claude subscription, as described in the
   setting's README.

### Quick Start

Online text classification:

```bash
cd reference_examples/text_classification
CONFIRM_COST=1 bash scripts/run_task_coevolve_20pct.sh
```

Terminal-Bench 2.1 (GPT-5.6-Luna):

```bash
cd reference_examples/terminal_bench_2
bash scripts/gpt56luna/collect_baselines.sh
CONFIRM_COST=1 bash scripts/gpt56luna/run_task_coevolve_20pct.sh
```

The naive and full-search arms are `run_naive_20pct.sh` and
`run_full_search.sh` next to each of these scripts.

## Citation

```bibtex
@article{miyai2026taskcoevolve,
  title   = {Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection},
  author  = {Miyai, Atsuyuki and Aizawa, Kiyoharu and Yamasaki, Toshihiko},
  journal = {arXiv preprint arxiv:2608.20169},
  year    = {2026}
}
```

## Acknowledgement

This code builds on [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness)
(MIT License). Its license is reproduced in
[`LICENSE-meta-harness`](LICENSE-meta-harness).

## Related Work from Our Group

**[Jr. AI Scientist and Its Risk Report](https://arxiv.org/abs/2511.04583)** — This paper comprehensively discusses the capability limitations of current state-of-the-art AI Scientists and their associated risks, which helps clarify the direction of our future work.

```bibtex
@article{miyai2026jr,
  title   = {Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper},
  author  = {Miyai, Atsuyuki and Toyooka, Mashiro and Otonari, Takashi and Zhao, Zaiying and Aizawa, Kiyoharu},
  journal = {TMLR},
  year    = {2026}
}
```

**[Paper Reconstruction Evaluation (PaperRecon)](https://arxiv.org/abs/2604.01128)** — This paper introduces the first systematic evaluation framework for quantifying the quality and risks of papers written by modern coding agents. In PaperRecon, an overview is created from an existing paper, after which an agent generates a full paper based on the overview and minimal additional resources, and the result is subsequently compared against the original paper.

```bibtex
@article{miyai2026paper,
  title   = {Paper Reconstruction Evaluation: Evaluating Presentation and Hallucination in AI-written Papers},
  author  = {Miyai, Atsuyuki and Toyooka, Mashiro and Zhao, Zaiying and Watanabe, Kenta and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
  journal = {arXiv preprint arXiv:2604.01128},
  year    = {2026}
}
```
