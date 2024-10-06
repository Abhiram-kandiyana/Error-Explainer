# Error-Explainer
The efficiency of debugging is critically impacted by the quality of error log messages. The compiler
messages are mostly vague and sometimes misleading which leaves the programmer clueless about the
source of error. Most beginners spend hours trying to find and fix a bug due to this issue. This project explores
the application of large language models (LLMs) for enhancing debugging productivity through clearer and
more actionable error explanations. Utilizing the pretrained Code-LLaMA model, known for its adeptness in
programming and coding tasks, we investigate two primary approaches to improve explanation quality: finetuning and prompting strategies.

Keywords: LLMs, few-shot prompting, Text Alignment, Code-Llama, Low Rank Adaptation (LoRA)

<!---
## Why not ChatGPT?
Yes, you can simply use ChatGPT but:
1. It is close-sourced and as an engineer it becomes hard to understand, improve or evaluate it's performance
2. It is slow. GPT-4o  
-->

# Overview
Our study introduces a custom error and alignment dataset tailored to refine the baseline capabilities of the Code-LLaMA model, aiming to produce more relevant and accurate
error explanations. We assess the effectiveness of each strategy using Perplexity (PPL), BERT-score, and
comprehensive human evaluation metrics. These metrics evaluate the clarity, relevance, and actionability
of the explanations, contributing to an empirical understanding of how LLMs can be optimized to support
software developers in debugging tasks. Our findings not only shed light on the comparative strengths and
limitations of fine-tuning versus prompting within the context of error explanation but also propose a
framework using few prompt strategies for further enhancement of automated debugging assistance tools
in software development environments.

# Results

## Test Results
| Strategy                       | BERT-score | Perplexity (PPL) | Human-Eval |
|---------------------------------|------------|------------------|------------|
| Zero-shot (Baseline)            | 0.82       | 9.6              | 28.67      |
| Fine-Tuning                     | 0.78       | 9.4              | 40.33      |
| Random-Prompting 4-shot         | 0.88       | 4.9              | 32.67      |
| <mark>Same-Class-Prompting 4-shot</mark> | <mark>0.89</mark>   | <mark>4</mark>          | <mark>46.67</mark>  |
| Manual-Prompting 4-shot         | 0.39       | 2.7              | 18.33      |

## Human Evaluation Results
| Examiner  | Zero-shot (baseline) | Finetuned | Random-Prompting 4-shot | Same-class-Prompting 4-shot | Manual-Prompting 4-shot | GPT-4 |
|-----------|----------------------|-----------|-------------|-------------|-------------|-------|
| 1         | 21                   | 44        | 31          | 46          | 17          | 51    |
| 2 | 28                   | 34        | 35          | 47          | 17          | 49    |
| 3         | 37                   | 43        | 32          | 47          | 21          | 53    |
| **Average** | 28.67                | 40.33     | 32.67       | <mark>46.67 </mark>      | 18.33       | <mark>51</mark>   |

**NOTE**: To read more about the results, metrics and methodology, please go to [project report](./project report.pdf)
