## In Defense of Superintelligence

The latest episode of Cal Newport's _Deep Questions_ podcast is titled [The Case Against Superintelligence](https://www.thedeeplife.com/podcasts/episodes/ep-377-the-case-against-superintelligence/) (also available with video on [YouTube](https://www.youtube.com/watch?v=y0RI5CnoDvs&t=3881s&pp=ygULY2FsIG5ld3BvcnQ%3D)). In it, he responds to a [recent episode](https://www.nytimes.com/2025/10/15/opinion/ezra-klein-podcast-eliezer-yudkowsky.html) of _The Ezra Klein Show_ in which Klein interviews Eliezer Yudkowsky about the existential risk of artificial superintelligence (ASI).

Cal describes why he disagrees with both Yudkowsky's central argument, that the risk posed by artificial intelligence is existential, and several of his supporting claims, like the fact that artificial superintelligence is both possible and inevitable.

I agree with Cal on may topics - I'm a regular listener to his show and I've found his advice (e.g. [Deep Work](https://www.amazon.com/Deep-Work-Focused-Success-Distracted/dp/1455586692), [Slow Productivity](https://www.amazon.com/Slow-Productivity-Accomplishment-Without-Burnout/dp/0593544854)) practically useful for nearly a decade now - but AI appears to be an area where our opinions diverge. After listening to both Cal's take and the [original interview](https://www.youtube.com/watch?v=2Nn0-kAE5c0) to which he is responding, I thought I'd take a minute to lay out the points on which I think Cal is mistaken, and in the process clarify some of my own positions.

### Summary of Yudkowsky's Position

Cal spends the first ~dozen minutes of the episode summarizing Yudkowsky's argument for the existential risk posed by artificial superintelligence.

He does so by first describing two examples that Yudkowsky invokes to establish this fact:

- An example in which ChatGPT provided undesirable advice on how to commit suicide to a teenager
- An example in which GPT-o1 (preview) [pursued an unexpected strategy](https://cdn.openai.com/o1-system-card-20240917.pdf) to achieve its objectives in a capture-the-flag (CTF) competition (reward hacking)

Having established Yudkowsky's take on our inability to control current AI systems, Cal moves on to describing how Yudkowsky draws the connection between this failure of control and human extinction: via the "straightforward" (Yudkowsky's words) implications of extrapolating this trend to more powerful systems.

Here, Cal is careful to acknowledge some nuance in Yudkowsky's argument, namely the distinction between malice and mere indifference. He acknowledges that we need not worry that these systems will become spontaneously malicious as they grow more powerful (a la [Skynet](https://en.wikipedia.org/wiki/The_Terminator)). Instead, it is equally dangerous and far more likely that a small divergence in the "values" of these systems will lead to their abject indifference towards humans as they become more and more capable. 

In the original interview, Yudkowsky uses a well-worn analogy to illustrate this point, describing how a superintelligent AI's relationship to humans will be analogous to our relationship with ants. Most of us don't set out with the goal of killing ants on a daily basis. However, when we are pursing a goal, like building something outdoors, and the presence of ants impedes our ability to achieve it, we exterminate them without a second thought. Its not that we hate ants and want to destroy them; its merely that we think our ability to get things done is more important than the ants' right to exist.

Ultimately, Cal summarizes Yudkowsky's argument thus:

- We can't control AI in its current form
- AI will not care about humans sufficiently when it becomes more powerful
- We all die

I find this a reasonably-accurate, albeit simplified summarization of Yudkowsky's position. In Cal's defense, it is not a simple matter to derive this argument directly from the interview because of the somewhat-meandering nature of Klein's questions and Yudkowsky's at-times inscrutable answers. 

Following this summary, Cal begins to construct his counterarguments to Yudkowsky's thesis, beginning with the first claim - that current AI systems are difficult to control.

### Predictability and Control

To refute Yudkowsky's claim, Cal argues that current AI systems are not _uncontrollable_ but merely _unpredictable_. The core of his refutation is built on the [observation](https://youtu.be/y0RI5CnoDvs?si=ameRgRYKKk9ltoTF) that current AI systems do not possess the same types of internal psychological states that humans do:

> ...to say that these agents have minds of their own or alien goals or ideas that match with our ideas, that's not an accurate way to talk about them - there are no intentions, there are no plans. There's a word guesser that does nothing but try to win the game of guessing what word comes next. There's an agent, which is just a normal program that calls it a bunch of times to get a bunch of words in a row. We can't always predict what those words are going to be. 

His argument seems to be: AI systems cannot be beyond our control because they do not possess internal psychological states like goals, volition, plans, etc.

I disagree with this argument, and I think Yudkowsky himself does most of the work of defusing it in the original interview. When Klein [asks](https://youtu.be/2Nn0-kAE5c0?si=jjwSbr8v1FM1fLf6&t=1667) "Tell me how you think about the idea of what the AI wants" Yudkowsky [responds](https://youtu.be/2Nn0-kAE5c0?si=h5szoav6pULtiu7U&t=1672):

> The perspective I would take on it is _steering_ - how a system steers reality, how powerfully it can do that. ...it is much more straightforward to talk about a system as an engine that steers reality than it is to ask whether it internally psychologically wants things.

Yudkowsky invokes the example of an AI chess engine. We could ask the question of whether the engine has goals (or wants, experiences, plans etc.) that resemble our own, but more immediately relevant is its ability to steer the state of the chess board towards a position that is advantageous to its observable "goal" of wining the game. The question of whether the system _wants_ things is therefore irrelevant to the question of control.

Cal returns to the example of [GPT-o1 CTF reward hacking](https://cdn.openai.com/o1-system-card-20240917.pdf) to support his point. He argues that while the agent's behavior was unanticipated (because it is inherently unpredictable), it doesn't constitute loss of control of the system because there is no volition in evidence here. Instead, the model did exactly what it was trained to do, and parrotted back some well-known Docker administration troubleshooting steps in response to an unforeseen challenge.

I believe this example defeats his point, rather than making it. This is clearly an instance in which model alignment failed. The engineers and researchers at OpenAI no doubt went to great lengths to ensure that this type of reward hacking would not happen when o1 was deployed, yet it did occur despite these efforts. To me, this appears to be an explicit attempt to control the behavior of this system, and, as Yudkowsky describes, these attempts are currently prone to failure for a variety of reasons. 

On this point, the underlying source of the disagreement between these two may lie in the language used to describe these systems, rather than the problem of control itself. Cal is loathe to anthropomorphize AI, something that Yudkowsky is certainly guilty of in his interview with Klein. 

Regardless of the language we use to discuss it, a system doesn't need volition, goals, or some other internal experience to be capable of diverging from the goals of its creators in potentially-dangerous ways. The ability to steer reality in meaningful ways is all that is required. AI agents already meet this bar, or they will soon, so we should be concerned about whether or not they are subject to our control.

### The Inevitability of ASI

Beyond arguments for our inability to control AI, Cal also takes issue with Yudkowsky's (mostly implicit) claims regarding the inevitability of this technology. Cal cites the following [exchange](https://youtu.be/2Nn0-kAE5c0?si=P0eorwGiTUblCVSV&t=164) at the beginning of the interview as offering the only hint as to why Yudkowsky thinks ASI is feasible:

> _Klein_: So I wanted to start with something that you say early in the book: that this is not a technology that we craft, it's something that we grow.

> _Yudkowsky_: It's the difference between a planter and the plant that grows up within it. We craft the AI growing technology, and then the technology grows the AI.

Cal interprets this as a reflection of Yudkowsky's belief in the potency of [recursive self-improvement](https://www.lesswrong.com/w/recursive-self-improvement). While this is an important part of the story (one we'll return to in a moment), I believe Yudkowsky's intent is much more tame. He is merely responding to Klein's question about "crafting versus growing" by summarizing an uncontroversial fact about machine learning. We design an "AI growing technology" - an optimization process like gradient descent via backpropagation - and then turn this loose on a voluminous amount of data. The result is a system that exhibits many of the behaviors that we'd like it to have, but whose internals are inscrutable.

Far from being a comment on how we'll achieve ASI, Yudkowsky is actually pointing out how little we understand about the way this technology actually works.

**ASI and Recursive Self-Improvement**

Although I don't think it follows from the source material to which he is responding, I'll still contend with Cal's argument regarding the infeasibility of recursive self-improvement because I believe two crucial misunderstandings are in evidence. 

Shortly after introducing the concept of recursive self-improvement, Cal confidently debunks it, [stating](https://youtu.be/y0RI5CnoDvs?si=oee7zlbhrgzVq3Dj&t=2722): 

> Most computer scientists think [recursive self-improvement] is all nonsense. A word-guessing language model trained on human text is exceedingly unlikely... [to] produce code... that's better than what any human programmer can produce.

He goes on to elaborate, claiming that LLMs are incapable of generating code that implements a "successor" AI system unless such code is present in their pretraining set (such code is not present, by definition, because otherwise the successor would already exist). More generally, he argues that AI agents are limited to generating content that they've already encountered within their human-curated pretraining data, and therefore they are incapable of exhibiting the type of innovation necessary to make recursive self-improvement possible.

Cal cites a [tweet](https://x.com/chamath/status/1980004924458197472) to support this claim, explaining that declining code completions from "vibe coding" tools are evidence that AI programming agents are suitable for toys and demos, but not production software, let alone something completely novel like a next-generation artificial intelligence system. 

On its face, we should be skeptical of this evidence because (1) the results elide multiple popular coding agents (e.g. Codex, Claude Code) and (2) it contradicts a [growing](https://dora.dev/publications/) [body](https://survey.stackoverflow.co/2025) [of](https://leaddev.com/the-engineering-leadership-report-2025) [work](https://www.youtube.com/watch?v=tbDDYKRFjhk&t=591s) suggesting that real developers are experiencing real increases in productivity via AI-assisted engineering.

More importantly, this analysis misses two key points regarding how recursive self-improvement might be realized. The first is that LLMs are _not_ limited to generating precisely the content that is present in their pretraining data. While pretraining data quality has a huge impact on LLM capabilities, these models don't just memorize text - they effectively compress huge volumes of it by learning patterns, relationships, and structures in language and the concepts it encodes. With the proper context, its possible to prompt an LLM to produce something that has never existed before.

The second point follows naturally from this first: human intervention, like prompting, is still necessary for technological innovation in AI to proceed, at least for the time being. Cal assumes too much about how the recursive self-improvement process must begin, suggesting that today's coding agents must somehow bootstrap themselves to the point where they could write their own internal logic. In reality, gains in the rate of AI technological progress are occurring today because of our ability to leverage AI to increase our productivity. The technology is still recursively self-improving, albeit with human input, because advances in the technology allow us to make more rapid progress in research and engineering, in turn leading to further advances in the technology, in a classic positive-feedback pattern. 

**Failure of the Current Paradigm**

As another piece of evidence that, far from being inevitable, ASI is likely infeasible, Cal points to a general slowing down in the rate of progress in AI development. He summarizes his position, which he describes in detail in an earlier [New Yorker article](https://www.newyorker.com/culture/open-questions/what-if-ai-doesnt-get-much-better-than-this), as follows:

> Starting about... two years ago, the AI companies began to realize that simply making the underlying language models larger... wasn't getting giant leaps in their capabilities any more... Everyone tried to scale more, everyone failed.  

I don't disagree with Cal's observations about recent progress in language model technology and the current state of the AI industry, but going directly from this point to "ASI is technologically infeasible" is a non-sequitur. 

These trends suggest that the current paradigm centered around auto-regressive large language models, and the scaling laws that govern the relationship between input resources and their resulting capabilities, may be reaching its limits. Indeed, some technical experts in the field have been [predicting](https://www.youtube.com/watch?v=4__gg83s_Do&pp=ygUKeWFubiBsZWN1bg%3D%3D) this outcome since before these trends were evident. But this doesn't say anything about whether we ultimately can or will achieve ASI, merely the timeline on which this might occur.

Near the end of their interview, Klein and Yudkowsky have an exchange that corroborates exactly this point:

> _Klein_: What do you say to people who just don't really believe that superintelligence is that likely? There are many people who feel that the scaling model is slowing down already. That GPT-5 was not the jump they expected from what has come from before it. That when you think about the amount of energy, when you think about the GPUs, that all the things that would need to flow into this to make the kinds of superintelligent systems you fear, it is not coming out of this paradigm... What would you say to these them?

> _Yudkowsky_: I'd tell these Johnny-come-lately kids to get off my lawn... I first started to get really really worried about this in 2003. Nevermind large language models. Nevermind AlphaGo or AlphaZero. Deep learning was not a thing in 2003. Your leading AI methods were not neural networks; nobody could train neural networks effectively more than a few layers deep because of the exploding and vanishing gradients problem. That's what the world looked like back when I first said 'uh oh, superintelligence is coming' 

Cal interprets Yudkowsky's response as a means of shutting down further discussion. He claims Yudkowsky's point is that people can't question his authority on this subject because he's been talking about it for a long time - before it was popular to do so.

While Yudkowsky's response isn't as clear as it could be, it is still obvious within the context of the interview that this is decidedly _not_ his point. Rather, he is making the case that his predictions regarding the technological feasibility of superintelligence and the existential risk it poses are independent of the current AI paradigm. New paradigms come and go, and with them our estimates for the timeline on which ASI will arrive might fluctuate, but the belief that such technology is possible is grounded in deeper theoretical convictions that remain unmoved.

### Final Thoughts

I mentioned at the beginning of this post that I am used to agreeing with Cal's conclusions on various topics, and I think this is because I've come to trust the process of analysis that he uses to arrive at them. Here on the topic of superintelligence, though, he is missing some of his usual rigor. 

In this episode, I think he spends so much energy taking issue with the language Yudkowsky uses to describe current agentic AI systems that he misses a more subtle point. When it comes to the question of control, it doesn't matter if we can attribute goals, volition, planning, or any other internal psychological state to these systems; all that matters is their ability to meaningfully steer reality and whether they do so in ways that align with our own values.

Following this, Cal attacks ASI's technical feasibility through the avenue of recursive self-improvement. His two major points here regarding the potential of current and future coding agents and the overall stagnation of the current LLM-centric paradigm are accompanied by weak evidence and faulty logic.

Overall, I suspect his general feelings regarding the current state of the AI industry (which I think are well-calibrated) may be warping his perspective when it comes to both some of Yudkowsky's finer points and the long-term potential of this technology.
