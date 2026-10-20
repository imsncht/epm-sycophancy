# EPM Sycophancy

## Research overview

Large language models (LLMs) can exhibit sycophancy: aligning their responses with a user's
stated beliefs, preferences, or emotional framing rather than with objective correctness.
Traditional evaluations often test this as a static, single-turn behavior by measuring how
readily a model concedes to an explicitly false or biased claim.

This project proposes **absorption** as a subtler, conversational mechanism: a model's later
reasoning may be shaped by premises introduced earlier in a conversation, even when those
premises are never explicitly re-endorsed or re-examined. The proposal situates absorption
alongside alignment drift, elaboration drift, and perspective mirroring, and outlines a way to
separate it from baseline (parametric) sycophancy using context-length manipulation and
behavioral metrics, with internal mechanistic probing as an optional exploratory extension.

## Full research proposal

Read the [full research proposal](documentation/absorbtion_research_proposal.md) in the
[`documentation/`](documentation/) folder.