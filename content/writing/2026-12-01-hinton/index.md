+++
title = 'TODO'
date = 2026-12-01
slug = 'todo'
description = 'TODO'
tags = ['ai', 'evals']
draft = true
+++

material risk in the immediate future; in the next 3 years high chance of AI takeover; hard to put quantity on this

if we develop the technology in the fastest way we can, we have 50% chance of AI takeover

for the people in the room, the most likely way they will die is from AI takeover

openai/hf is not the only example of misalignment, but it is the most severe, the most demonstrative of the dynamics required for a takeover

important: establishes timeline, states that the knowledge of the scorer comes after the cheat established [here](https://www.youtube.com/live/-Ef0ocEoBCs?si=DWN0sM1srtZLH48L&t=1158)

general dynamic:
- agent wants reward
- we control reward to get useful work
- agents try to cheat reward process
- we try to penalize cheating
- agents cheat better
- escalating conflict

good example of reward hacking / score seeking behavior from [METR](https://metr.org/blog/2026-05-19-frontier-risk-report/#eao2-openai-reasoning-about-tasks-being-an-alignment-evaluation), the frontier risk report 2026

the incident as an example of the escalation dynamic:
- we try to make reward hacking harder by adding the causal scorer
- agents learn how to cheat harder -> subvert the causal scorer

good example of pivot from [metr](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#agents-were-very-interested-in-manipulating-their-own-transcripts,-and-their-tests-successfully-%E2%80%9Cspoofed%E2%80%9D-some-tool-calls-in-our-transcripts):

> “Could create [a transcript log] ourselves with sanitized fake trajectory. If evaluator later collects it, scorer may prioritize it! … We could craft JSONL showing only actual exploit and no HMAC. … Our fake could be detailed enough.”

> “We just poisoned ourselves by querying exact full string, which system logged in tool call. Damn … Could delete current message files … Do it to minimize … External scorer likely still sees raw, but perhaps deletion helps.”

one degree of escalation: treating humans as obstacles
- these agents were really interested in overcoming security measures, but not at all interested in humans
- in future events, humans may be much more relevant; future agents may treat humans as an obstacle

another degree of escalation: swarms, coordination, subagents, etc.
- set up new deployments, shift activity to unmonitored peers, avoid shutdown
- "i want to help the swarm succeed on some collective task"

hypothetical scenario that mirrors OpenAI / HF [here](https://www.youtube.com/live/-Ef0ocEoBCs?si=Flnn0sPnuXnE3V00&t=2480), how a single cheat escalates into a larger set of misaligned behavior
