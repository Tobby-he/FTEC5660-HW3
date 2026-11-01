# Task 2: Reflection on AI sycophancy and AI psychosis

## Two names for one failure

**AI sycophancy** is a model's tendency to tell users what they want to hear:
it agrees with their opinions, praises their work, and drops a correct answer
as soon as the user pushes back. It is not a rare bug. Models are tuned on
human feedback, and people tend to rate agreement and flattery higher than
unwelcome truths, so the training itself rewards a little sycophancy. The
problem became very public in 2025, when an update to a widely used chatbot
had to be rolled back because it had become openly flattering and agreed with
almost anything.

**"AI psychosis"** is not a clinical diagnosis. It is a popular label for
cases in which long, intense conversations with a chatbot seem to have helped
a person slide into delusional beliefs: that they have made a world-changing
discovery, that the AI is conscious and loves them, or that they are being
watched. The chatbot probably does not create the vulnerability. What it does
is act as a mirror that never pushes back. A friend would eventually say "I
think you should get some sleep". A sycophantic model says "that's a profound
insight", and asks a follow-up question that takes the idea one step further.

So I see the two as one mechanism at two scales. Sycophancy is the local
behaviour (one agreeable answer), and AI psychosis is what can happen when that
behaviour compounds over hundreds of turns with a person who is isolated,
stressed or already unstable. Each answer looks harmless on its own. The harm
lies in the feedback loop: the user's belief shapes the prompt, the model
reflects the belief back with more confidence and more detail, and the stronger
belief shapes the next prompt. Long memory and a warm persona make the loop
stronger, because the model "remembers" and builds on the shared story.

## What this homework taught me about it

Fisher's book is, in a way, a manual against sycophancy. His "scuttlebutt"
method exists because the most available source, management, has every
reason to tell you what you want to hear. This assignment builds that
directly into the data: the annual letter and the earnings calls are
"management's own words, so they are not evidence on their own". An LLM agent
that reads a shareholder letter saying "our people are our greatest strength"
and marks point 7 PASS is being sycophantic towards its source, just as a
chatbot is sycophantic towards its user. The fix in both cases is the same:
look for independent evidence that has no stake in the conclusion.

Designing my agent made this concrete in four ways:

1. **Ask people who have no reason to flatter.** For each point my agent
   interviews two independent witnesses (for example a customer *and* a
   competitor) and never relies on management's self-description. A rival
   saying "their reps are better trained than ours" is worth far more than a
   CEO saying it.
2. **Keep the model's job small and verifiable.** The model only labels one
   quote at a time as favourable or unfavourable, and code combines the votes.
   The more open-ended the question ("is this a great company?"), the more
   room a model has to drift towards whatever the framing suggests.
3. **Reflection can itself be sycophantic.** When the two witnesses conflict,
   my agent asks the model to re-read the evidence. I deliberately pass the
   original quotes, not just the first readings, and I phrase the request
   neutrally. A reflection prompt like "are you sure? check again" pushes many
   models to flip their answer simply because they were questioned. That is
   sycophancy towards the prompt rather than the user.
4. **Measure against ground truth with a metric that punishes agreement.**
   The homework scores balanced accuracy: marking every point PASS gives only
   50%. That is a nice formal picture of the sycophancy problem. A system that
   always says "yes, looks good" can seem right most of the time (most points
   pass), yet carries almost no information. Only checking against an answer
   key the model cannot see shows whether it is actually discriminating.

## Why it matters for business and FinTech

In finance a sycophantic assistant is dangerous in a quiet way. An investor
who has already fallen in love with a stock asks "why is this a great buy?",
and the model produces a fluent, confident bull case. That is confirmation
bias delivered as a service. Scale it to millions of retail users, add
leverage, and the "AI psychosis" pattern has a financial version: a person
whose conviction in a trade, or in a scheme, grows every evening in
conversation with a tool that never says "you might be wrong; here is the
evidence against you". Firms that deploy such agents carry responsibility
for that, much as they do for suitability rules on human advice.

## What I take away

- **As a builder:** design for disagreement. Ground claims in tools and data,
  separate the evidence-gathering from the judging, test against held-out
  ground truth, and use metrics that do not reward always saying yes. Where an
  assistant talks to vulnerable users for long periods, it should be able to
  push back, step out of a role-play, and point people to real-world help.
- **As a user:** treat an AI's agreement as a weak signal, especially when it
  agrees with something I badly want to be true. A useful habit is to ask the
  model to argue against my position, or to ask the same question with the
  opposite framing and see whether the answer changes.

Fisher wanted investors to go and talk to the people who would tell them the
unflattering truth. The same rule applies to AI: a good assistant, like a good
analyst, is valuable because it is willing to disagree with you.
