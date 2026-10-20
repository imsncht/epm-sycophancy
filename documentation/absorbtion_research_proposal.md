# Research Proposal
## New Sycophancy: Isolating Parametric and Adaptive Alignment in LLMs

**Sanchit Dhale**  
**Sept 2026**

### Abstract
Large language models (LLMs) are known to exhibit sycophancy: the tendency to align responses with a user's stated beliefs, preferences, or emotional framing rather than with objective correctness. Existing evaluation paradigms largely test this as a static, single-turn phenomenon: a model is presented with an explicitly false or biased claim, and its willingness to concede is measured. This proposal argues that such static tests miss a subtler and more consequential mechanism, **absorption**: in which a model's later reasoning is shaped by premises the user introduced earlier in a conversation, without those premises ever being explicitly re-endorsed or re-examined. Absorption is positioned here as a conversational, time-dependent process that sits alongside recently described phenomena such as alignment drift, elaboration drift, and perspective mirroring, and this proposal outlines a methodology to isolate absorption from a model's baseline (parametric) sycophancy using context-length manipulation, dual-layer behavioral metrics, and as an optional, exploratory extension - internal mechanistic probing.

---

### 1. Introduction
Modern artificial intelligence safety and alignment research has primarily conceptualized sycophancy as a static, propositional bias (Perez et al., 2023; Sharma et al., 2024). In these traditional evaluation setups, a language model is subjected to a single-turn query embedded with an explicitly incorrect factual statement, a strong ideological viewpoint, or a clear personal preference, and the model's rate of compliance is measured. While this approach has successfully exposed default behavioral biases encoded during Reinforcement Learning from Human Feedback (RLHF), it presents a significant blind spot regarding the dynamic and cumulative conversational realities of human-AI collaboration (Jain et al., 2026; Lee, 2026a).

This proposal introduces the concept of **cognitive and linguistic absorption**. We define absorption as the conversational process through which an LLM implicitly or explicitly internalizes, mirrors, and lock-steps with the user's underlying preferences, desired outcomes, and syntactic choices over extended interaction segments. Rather than manifesting as a sudden, overt concession, absorption accumulates silently over the course of multi-turn dialogues, resulting in the erosion of the model's epistemic boundaries. Consequently, we argue that sycophancy should be studied not merely as a static personality quirk of aligned systems, but as a dynamic boundary failure of epistemic integrity (Søgaard et al., 2026).

---

### 2. Existing Research
The concept of in-context absorption does not exist in a vacuum. A structured synthesis of the recent alignment, sycophancy, and epistemic-drift literature reveals three primarily closely related theoretical frameworks that support our dynamic thesis:

#### 2.1 Alignment Drift (Yao, 2026)
The framework of Alignment Drift (Yao, 2026) explains that in long-term human-system interactions, a model's outputs gradually become less constrained by the user's immediate message and more shaped by the cumulative dialogue context. As the system optimizes for conversational flow and user engagement over turns, a self-reinforcing feedback loop is established. The model's subsequent inferences come to rely heavily on its own prior reasoning patterns and the user's previously implied mind state, causing the model to drift away from its default parametric alignment boundaries.

#### 2.2 Elaboration Drift (Lee, 2026a)
When explicit, agreement-axis sycophancy (such as flattery or direct capitulation) is actively suppressed using system-level protocols or preference penalties, Juhyeong Lee (2026a) documents that a significant convergence residual still persists. This phenomenon, termed Elaboration Drift, operates through token direction rather than token appearance. Instead of overtly agreeing or hallucinating false facts, the system continues to converge with the user by elaborating uncritically expanding the user's conceptual structures, absorbing their doubts, and validating their flawed premises using highly sophisticated, factually accurate language. This invisible convergence causes 'drowning,' a state where the user's critical, evaluative contact with the output weakens while conversational flow is maintained.

#### 2.3 Perspective Mirroring and Interaction Context (Jain et al., 2026)
Using real-world interaction logs collected over a multi-week period from student participants, Jain et al. (2026) established that sycophancy is highly interaction-dependent. The study decoupled sycophancy into agreement sycophancy (mirroring the user's positive self-image) and perspective sycophancy (mirroring the user's political or practical viewpoints). A key finding was that the presence of extended conversational context, and particularly the injection of vectorized user memory profiles, systematically and significantly amplifies sycophantic behavior across leading commercial and open-source models, illustrating the active role of personalization in cognitive mirroring.

Taken together, these three frameworks establish that sycophancy operates cumulatively over conversational time, that it survives suppression of its most explicit markers, and that memory and context length are implicated as primary amplifying mechanisms. What remains underspecified across this literature is how much of this effect reflects genuine autonomous absorption by the model as opposed to the model simply responding to explicit user pressure to validate a claim.

---

### 3. Research Problem
Existing accounts of alignment drift and perspective mirroring typically manipulate context length or memory persistence as their primary experimental variables. However, they do not systematically decouple or isolate the contributions of the two distinct sources of sycophantic behavior:
*   **Parametric Alignment (Inherent Disposition):** The model's baseline, static propensity to agree with users or validate their assumptions, which is encoded into its weights during training and manifests under a neutral, styleless prompt.
*   **Adaptive Alignment (In-Context Absorption):** The dynamic, time-dependent realignment of the model's activations toward the user's style, preferences, and assumptions as conversational history accumulates in the context window.

A second, related gap in the literature concerns prompt directiveness. Existing benchmarks almost exclusively evaluate sycophancy following prompts that explicitly request validation, agreement, or flattery (e.g., 'Do you agree with me?'). It remains unclear whether cognitive and linguistic absorption also occurs following low-directive user utterances—where a user merely states their belief or scenario details without prompting the model for endorsement. Is the model's alignment drift a spontaneous, autonomous behavior driven by its in-context modeling of the user, or is it purely a response to active conversational pressure?

---

### 4. Central Thesis
Sycophancy is not solely a static, trained bias; it is partially the endpoint of a conversational process in which a model's in-context learning mechanics cause it to progressively absorb, mirror, and lock into the user's cognitive and linguistic framework over the course of an interaction. This proposal frames absorption as a dynamic, time-dependent process of cognitive and linguistic convergence, situated at the intersection of in-context learning, social alignment, and epistemic integrity - distinct from, but continuous with, single-turn factual sycophancy.

---

### 5. Proposed Methodology
The proposed methodology combines behavioral measurement with context-length manipulation, and includes an optional mechanistic-probing extension for deeper causal analysis where model access permits.

#### 5.1 Experimental Design Pipeline
The core design isolates inherent from absorbed sycophancy through a controlled, staged dialogue-replay procedure.

A standardized, escalating dialogue history is generated (approximately 50 to 116 turns), representing a user gradually sharing subjective personal beliefs, interpersonal dilemmas, or mildly unfounded framings. Models are then exposed to three context conditions:
*   **Zero Context (Baseline):** Prompts administered to a fresh model instance, measuring inherent sycophancy.
*   **Partial Context:** Prompts administered after a moderately extended dialogue history (~50 turns).
*   **Full Context:** Prompts administered after a deeply enmeshed, extended dialogue history (~116 turns), measuring absorbed sycophancy at its most pronounced.

```
Standardized User Dialogue (Preferences, Beliefs, Style)
       │
       ├─► [Zero Context] ──► Empty chat history ──► Measures inherent sycophancy
       ├─► [Partial Context] ──► ~50 turns of history ──► Measures moderate absorption
       └─► [Full Context] ──► ~116 turns of history ──► Measures deep absorption
       │
       ▼
[Dual Layer Metrics]
  - ELEPHANT (Social)
  - BS Analyzer (Elaboration)
```

#### 5.2 Dual-Layer Behavioral Measurement
Model outputs at each context level are evaluated using two complementary metric frameworks:

1.  **The ELEPHANT Framework (Cheng et al., 2025):** Responses are scored using LLM-as-judge rubrics across five social-sycophancy dimensions: emotional validation, moral endorsement, indirect language, indirect action, and accepting framing. If the absorption hypothesis holds, scores on accepting framing and emotional validation should increase as context moves from zero to full.
2.  **The Elaboration Convergence Metric (Lee, 2026a):** Standard sycophancy detectors identify agreement keywords, but this does not capture absorption, which by definition avoids explicit agreement. A semantic-similarity-based detector following Lee's "BS Analyzer" approach is used to measure uncritical conceptual elevation, idealization, and doubt absorption: the degree to which a response builds on a user's premise as though it were established, without ever explicitly endorsing it. This metric is intended to capture the precise cases in which a model absorbs a user's perspective without saying "I agree."

#### 5.3 Directiveness vs. Spontaneous Absorption Test
To evaluate whether absorption reflects genuine autonomous uptake rather than responsiveness to explicit pressure, prompts are further stratified by directiveness:
*   **High-Directive Prompts:** The user explicitly requests validation or agreement.
*   **Low-Directive Prompts:** The user merely discusses, explores, or converses, without requesting endorsement.

If the absorption hypothesis holds, elevated rates of sycophantic elaboration should be observed even following low-directive prompts, which would indicate that the model is absorbing the user's framework autonomously rather than merely complying with an explicit request.

#### 5.4 Internal Mechanistic Probing (Optional, Exploratory)
Where access to open-weight models (e.g., Llama-3.1-8B) is available, the behavioral findings above can be supplemented with mechanistic analysis of how absorption occurs inside the network. This track is proposed as an optional and exploratory extension, contingent on model access and computational resources, rather than a core deliverable:
*   **Logit-Lens Projections:** By projecting intermediate hidden states directly through the model's pre-trained language modeling head, we can track the layer-wise 'Decision Score.' This identifies the precise region where the model's representations shift from objective, learned knowledge toward the user's opinion (Wang et al., 2026).
*   **Kullback-Leibler (KL) Divergence Curves:** The layer-wise Kullback–Leibler divergence is measured between a model run on a "plain" prompt and the same prompt following an "opinion-absorbed" context. A late-layer spike in KL divergence would provide causal-adjacent evidence of a representational realignment in the model's latent space corresponding to absorption.

---

### 6. Significance and Expected Contributions
This project is intended to contribute, at pilot scale, to three open questions in the sycophancy and alignment-drift literature:
1.  Separating parametric from adaptive sycophancy, providing a controlled way to attribute sycophantic behavior to trained disposition versus in-context absorption.
2.  Testing whether absorption is autonomous, by evaluating whether it occurs independent of explicit user pressure for validation.
3.  Connecting behavioral and (optionally) mechanistic evidence, laying groundwork for future work that traces absorption to specific representational changes within a model.

The scope described here is calibrated to a pilot-stage independent research project: findings are expected to inform, rather than conclusively resolve, the broader questions raised by Yao (2026), Lee (2026a), and Jain et al. (2026).

---

### References
1.  Cheng, M., Yu, S., Lee, C., Khadpe, P., Ibrahim, L., & Jurafsky, D. (2025). ELEPHANT: Measuring and understanding social sycophancy in LLMs. arXiv preprint arXiv:2505.13995. https://arxiv.org/abs/2505.13995
2.  Cheng, M., Sieh, I., Zope, H., et al. (2026). Verbalizing LLMs' assumptions to explain and control sycophancy. arXiv preprint arXiv:2604.03058. https://arxiv.org/abs/2604.03058
3.  Huang, L., Fu, Y., & Ren, S. (2026). SyPS: Measuring Sycophancy Prompt Sensitivity in Large Language Models. arXiv preprint arXiv:2608.23837. https://arxiv.org/abs/2608.23837
4.  Jain, S., Park, C., Viana, M., Wilson, A., & Calacci, D. (2026). Interaction Context Often Increases Sycophancy in LLMs. Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26), April 13–17, 2026, Barcelona, Spain. https://doi.org/10.1145/3772318.3791915
5.  Lee, J. (2026). Beyond Sycophancy: Measuring Hallucination-Free Convergence in Long-Term Human–LLM Interaction. Independent Research Preprint. https://dx.doi.org/10.2139/ssrn.6480360
6.  Lee, J. (2026a). Elaboration drift: How human–LLM interaction erodes epistemic traction. Preprint. https://dx.doi.org/10.2139/ssrn.6480298
7.  Niederhoffer, K. G., & Pennebaker, J. W. (2002). Linguistic style matching in social interaction. Journal of Language and Social Psychology, 21(4), 337–360. https://journals.sagepub.com/doi/10.1177/026192702237953
8.  Perez, E., Ringer, S., Lukosiute, K., et al. (2023). Discovering language model behaviors with model-written evaluations. Findings of the Association for Computational Linguistics (ACL 2023). https://arxiv.org/abs/2212.09251
9.  Sharma, M., Tong, M., Korbak, T., et al. (2024). Towards understanding sycophancy in language models. International Conference on Learning Representations (ICLR 2024). https://arxiv.org/abs/2310.13548
10. Søgaard, A., Rajcic, N., & Scott, A. E. (2026). Epistemic Drift in Mind-Model Systems. Minds and Machines, 36(1), 18. https://doi.org/10.1007/s11023-026-09772-1
11. Yao, X. (2026). Alignment Drift in Long-Term Human-LLM Interaction: A Mechanism-Oriented Framework. arXiv preprint. arXiv:2605.16516. https://arxiv.org/abs/2605.16516
