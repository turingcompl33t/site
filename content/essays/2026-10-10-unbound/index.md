+++
title = 'Opaque CoT Makes Model-Intrinsic Properties Even More Important'
date = 2026-10-10
slug = 'unbound-opaque-cot'
description = 'An addendum to an article I recently published in light of new results regarding GPT-6 Astra'
tags = ['ai', 'alignment' 'evals']
+++

I recently published an article in [Gray Space](https://www.lineofdeparture.army.mil/Journals/Gray-Space/Archive/2026-Edition/Agents-Unbound/) responding to the [OpenAI / Hugging Face incident](https://openai.com/index/hugging-face-incident-and-the-road-ahead/). I make the argument that the type of misalignment we saw in this incident is incompatible with military cyber operations, despite the level of capabilities that the agents in question exhibited. I go on to say that we should address this by navigating towards more desirable intrinsic model properties like _boundedness_--the propensity to identify and hold to an (often implicit) authorization boundary.

Despite the military/cyber coloring, this is really just a call for more post-training alignment work, and a new class of cyber evals intended to assess alignment in cyber-relevant contexts, the goal being to reduce the probability that agents pursue [unhinged instrumental goals](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).

A fair critique of this argument goes something like: "we don't need aligned models if we can monitor them in real time and thereby prevent them from doing heinous stuff." I respond to one form of this argument in the article, stating that cyber classifiers and refusals will not solve this problem because we can't realistically levy general cyber constraints on agents that we intend to conduct cyber operations.

I fail to respond, however, to stronger version of the "monitoring" argument. This one advocates the use of an external, LLM-based judge to monitor the agent's activity and determine whether it has drifted out-of-scope. Indeed, this is an approach that researchers from both [AISI](https://www.aisi.gov.uk/blog/building-a-more-secure-environment-for-evaluating-dangerous-capabilities) and [METR](https://metr.org/notes/2026-09-27-implementing-a-basic-blocking-action-monitor/) appear to be integrating with their evaluation infrastructure. In principle, there isn't any reason why the same architecture couldn't be applied outside of the lab, in the context of a cyber operation, to prevent unsanctioned activity, scope violations, etc.

## TL;DR

- We adop

## References

- My original article in Gray Space, [Agents Unbound: Autonomous Military Cyber Operations Require More than Just Capability](https://www.lineofdeparture.army.mil/Journals/Gray-Space/Archive/2026-Edition/Agents-Unbound/)
- OpenAI's final [writeup](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) on the incident, with links to other artifacts including their [technical report](https://cdn.openai.com/pdf/67869394-cb91-4c12-888c-5cbd85c7814c/OpenAI-Hugging-Face%20Incident-Technical-Report.pdf)
- The [METR-Redwood report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) which establishes the motivations of the agent swarm behind the incident
