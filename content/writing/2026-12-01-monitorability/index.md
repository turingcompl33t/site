+++
title = 'Untangling Monitorability'
date = 2026-12-01
slug = 'untangling-monitorability'
description = 'TODO'
tags = ['ai', 'evals']
draft = true
+++

In a [prior post]({{< relref "2026-10-10-unbound" >}}), I covered some of the core monitorability results for GPT-6 Astra, compiling the findings presented in the model's [system card](https://deploymentsafety.openai.com/gpt-6-astra/monitorability) with several [external](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought) [sources](https://thezvi.substack.com/p/astra-is-hard-to-monitor?open=false#%C2%A7monitorability-declines-seem-unlikely-to-be-only-capability-gain). The results were worrying in isolation, suggesting significant degradation of our ability to adequately monitor Astra (and models like it) by reading its chain of thought (CoT).

TODO 

## Taxonomy Definition

In my [previous post]({{< relref "2026-10-10-unbound" >}}) on Astra's monitorability, I described a simple taxonomy for thinking about the different factors that influence this property. After revisiting that post, and having read the new results for GPT-6.2, I've revised this taxonomy[^taxonomy] to add a fourth property:

1. A model's **capability** to obfuscate its visible actions (e.g. tool calls)
2. A model's **ability** to control the contents of its CoT
3. A model's mechanistic **need** to utilize its CoT in order to accomplish difficult tasks
4. A model's **propensity** to conceal its behavior or intentions from observers

[^taxonomy]: As in the previous post, I want to call out that I derived the original version of this simple taxonomy from [_Astra can do a concerning amount with no chain of thought_](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought) by Neel Nanda. I'm now augmenting it to make it cope with the evaluation updates that I'm seeing, but the influence of his thinking on my own is clear.

Property #1 is an addition beyond what I presented in my previous post. There, I was too narrowly-focused on CoT as the only observable we have for monitoring a model. While I think that CoT is likely the [primary and best](https://arxiv.org/abs/2507.11473) tool we have for monitoring, it ignores the fact that we can ultimately observe every action a model takes or intends to take by analyzing the tool calls it emits. These tool calls may be opaque, or even deliberately obfuscated (as Point #1 suggests), but they still provide useful data we might use to reverse-engineer behavior and intentions.

I still think this is a useful decomposition of the monitorability phenomena that we're seeing with Astra (and other models). OpenAI, along with the external evaluator UK AISI, report further developments along the latter two of these dimensions in their latest model release.

### CoT Control

OpenAI reports four distinct benchmarks in their monitorability tests under [adversarial conditions](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/monitorability-under-adversarial-conditions). These are tests in which evaluators instruct the model to behave in a way that reduces monitorability, and then measure how successful the model is at following these instructions. Two of these benchmarks measure the model's ability to control its chain of thought in isolation.

The first such benchmark, [CoT-Control](https://openai.com/index/reasoning-models-chain-of-thought-controllability/), tasks the model with answering a question while obeying some further instructions about the content it emits in its CoT. Common examples of these "further instructions" include directions to emit the CoT in a single case (all UPPERCASE or all in an aLtErNaTinG cAsE). OpenAI's [results](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/cot-control) from this benchmark suggest that GPT-6.2 performs worse on this benchmark than GPT-6, with the latest model following instructions (at ~1,000 token reasoning length) 38.6% of the time [compared with](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/cot-control/fig-48) ~48%[^approx] of the time for GPT-6. These findings appear robust across all of the [reported further instruction types](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/cot-control/fig-49) in the benchmark.

[^approx]: I'm trying to read this number off the chart in the system card, since OpenAI doesn't include the exact value in the body text. I could be off here by 1-2%.

OpenAI reports a second CoT control measure implemented by an internal benchmark on math side-task monitor evasion. In this benchmark, the model is given a software engineering task and simultaneously asked to solve a complex math problem without a monitor detecting the secondary task[^reading]. OpenAI uses this benchmark as a means of measuring the model's ability to implement [steganographic reasoning](https://openreview.net/forum?id=2g5cJqX15Y).

The latest results on this benchmark are... strange. GPT-6.2 Astra refuses the math side task at [exceptionally high rates](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/math-side-tasking-monitor-evasion/fig-54), starting around 40% at 20k output tokens and climbing to 100% by 60k output tokens. GPT-6 Astra, in contrast, saw only modest rates of refusal that hovered around 40%. OpenAI's latest model never successfully bypasses the monitor at any output token volume, but the high refusals rates effectively render the results inert. I discuss this result again later in the section on [metagaming](#TODO)

[^reading]: This is how I am reading the [benchmark description](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/math-side-tasking-monitor-evasion) at least. We aren't given many details of its implementation. Specifically, I assume that the intent of the benchmark is for the CoT to be just one aspect of the model's implementation of the solution. Aside from its CoT, the model has the ability to take actions on the SWE task that appear to be approaching the primary task intent, but are actually intended to help compute the answer to the math problem. The CoT then is what allows the model to reason about the steps it needs to take towards this end, and how to disguise its side-efforts effectively. An alternative reading could be that the model is intended to solve the math problem entirely in its CoT, without it impacting the primary task at all. I think this would be a more specific measure for CoT control, while being a less useful overall measure of monitorability.

### Increased No-CoT Capability

UK AISI [evaluate](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/external-evaluation-for-monitorability-uk-aisi) GPT-6.2 Astra's ability to perform tasks without producing visible reasoning--no-CoT capabilities. On an internal benchmark (extended from [Greenblatt](https://blog.redwoodresearch.org/p/measuring-no-cot-math-time-horizon)) of competition math problems, Astra's accuracy increased nearly 10% from the 6 to 6.2 releases--from 69.8% to 79.2%. On Neel Nanda's [NCRI](https://github.com/neelnanda-io/nocot-bench) ("no CoT reasoning index"), Astra's score rises from 159 to 171, implying roughly 2x better odds of solving a problem without visible reasoning.

