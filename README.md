# Simulating Human Memory with Language Models

This repository contains data and code for the paper, [Simulating Human Memory with Language Models](https://arxiv.org/abs/2605.25680).

> Abstract: Language models are increasingly being deployed as user simulators, but their memory is far more reliable than that of real users. To measure this gap, we run a series of classic memory experiments from psychology on both humans and language models. Across tasks, we find that out-of-the-box language models exhibit better memory than humans, even when prompted to imitate human behavior. We then show that better prompting strategies and the use of a compactor can cause language models to forget content in a more human-like way.  Using these methods, we show preliminary evidence that language models with human-like memory constraints can function as more effective user simulators in a downstream education task. Finally, we release human reference data and benchmarks to support future work on simulating human memory with language models.

## Outline
The codebase is organized as follows: 
 - Code for the ten memory tasks is in `bench`
 - Human data and model outputs are in `runs`
 - Code for computing scores is available in `src`
 - Code for the educational document reranking experiment is in `application`
 - Additional data for running tasks is in `data`

**Note 5/24/2026**: We are currently developing additional tooling to make it easier to run our benchmark and compute human-model similarity scores on new models. Stay tuned! 

## Citation
```
@misc{wang2026simulatinghumanmemorylanguage,
      title={Simulating Human Memory with Language Models}, 
      author={Qihan Wang and Nicholas Tomlin and Michael Hu and Brian Dillon and Tal Linzen},
      year={2026},
      eprint={2605.25680},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.25680}, 
}
```
