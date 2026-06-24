Learning Faster with Better Tokens: Parameter-Efficient Vocabulary Adaptation for Specialized Text Summarization

Abstract

Large language models pretrained on general-domain corpora often exhibit tokenization inefficiencies when applied to specialized do-mains. Although continual pretraining for domain adaptation partially alleviate performance degradation, it does not resolve the fundamental vocabulary mismatch. To address this gap, we introduce a targeted parameter-efficient domain adaptation approach that combines vocabulary adaptation with pretraining for LLM-based text summarization. Our unified framework augments pretrained tokeniz-ers with domain-specific tokens while selectively replacing under-trained and unreachable tokens to limit parameter growth. We evaluate our approach on Llama-3.1-8B and Qwen2.5-7B across legal and medical summarization tasks on a challenge-oriented evaluation protocol focused on expert-driven text and summaries which typically has higher concentration of over-fragmented Out-of-Vocabulary (OOV) words. The vocabulary adaptation algorithm enhances the overall quality of the summarization model by improving semantic similarity between the generated summaries and their references. In addition, the adapted model produces summaries that incorporate more appropriate novel and domain-specific words, leading to improved coherence, relevance, and faithfulness. We further observe that our proposed approach significantly reduce training time by 35 − 55% over continual pretraining and reduce parameter counts up to 37% w.r.t expansion-only methods. We make the code-base publicly available at https://github. com/gb-kgp/VocabReplace-Then-Expand.

Introduction

While large language models (LLMs) have revolutionized natural language processing, adapting gen-

This is the author’s version of the manuscript. It is posted here for your personal use. Not for redistribution. To Appear in the the 64th Annual Meeting of the Association for Computational Linguistics, ACL (Mains) 2026.

eralist models to expert domains remains challenging due to high vocabulary mismatch between general and domain-specific corpora. Recent domain-specific models including Meditron-70B (Chen et al., 2023); BioMistral (Labrak et al., 2024), built on Mistral-7B and further pretrained on PubMed Central; and PMC-LLaMA (Wu et al., 2024) demonstrate that continued pretraining on specialized corpora yields substantial performance im-provements. However, vocabulary mismatch fundamentally limits these gains: PubMedBERT (Gu et al., 2021) demonstrates that medical terms like "naloxone" fragment into meaningless subwords ("nal", "##ox", "##one"), while domain-specific vocabularies treat them atomically. This tokeniza-tion inefficiency imposes substantial costs—non-English and domain-specific text can require up to 13× more tokens than English (Rust et al., 2021; Ahia et al., 2023; Petrov et al., 2023), directly increasing API costs, latency, and memory require-ments. Recent work establishes that this fragmentation reduces effective context window size and impedes learning meaningful representations (Hof-mann et al., 2022; Kaplan et al., 2025).

The conventional approach to addressing vocabulary mismatch involves domain-adaptive pre-training (DAPT), where models undergo continued pretraining on domain-specific corpora (Gu-rurangan et al., 2020). While effective, this paradigm presents significant practical limitations. BioMistral-7B required 32 A100 GPUs for 20 hours on 3 billion tokens from PubMed Central, while Meditron-70B consumed 128 A100 GPUs for 332 hours processing 46 billion tokens while achieving marginal improvements. For contemporary large language models, Hu et al. (2022) note that full fine-tuning is “prohibitively expen-sive”, requiring complete parameter updates and storage of separate model instances per domain. While parameter-efficient methods reduce trainable parameters, they do not address the underlying tokenization inefficiency–vocabulary mismatch.

An alternative paradigm that directly addresses vocabulary mismatch is vocabulary adaptation, modifying a pretrained model’s tokenizer and embedding layer to incorporate domain-specific vo-cabulary. Recent works (Sachidananda et al., 2021; Hong et al., 2021; Liu et al., 2023; Yamaguchi et al., 2024; Balde et al., 2024; Gao et al., 2024; Balde et al., 2025) establishes this as a resource-efficient path. However, vocabulary expansion introduces computational overhead through parameter growth: adding 10,000 tokens to Llama-3-8B requires approximately 80 million additional parameters (at 4096-dimensional embeddings), representing non-trivial increase in model size and inference cost. Land and Bartolo (2024) reveal a critical insight: contemporary language models contain 0.1 − 1% severely under-trained "glitch to-kens"—vocabulary tokens that occupy vocabulary slots but contribute minimally due to insufficient pretraining exposure. This observation suggests efficient vocabulary adaptation is possible by strategically replacing under-trained tokens with domain-specific vocabulary, achieving adaptation benefits with minimal parameter expansion (Purason et al., 2026).

In this work, we propose a vocabulary adaptation method that strategically replaces under-trained and unreachable tokens with domain-specific vocabulary before resorting to expansion, thereby minimizing parameter overhead while enabling effective domain specialization. Our approach operates on Llama-3.1-8B and Qwen2.5-7B and consists of four key steps: (1) we train a BPE tok-enizer on domain-specific corpora to identify candidate domain vocabulary, (2) we select the top 10,000 tokens based on frequency and coverage statistics as our vocabulary adaptation budget, (3) we compile a replacement candidate list by identifying under-trained and unreachable tokens using Land and Bartolo (2024) methodology, and (4) we replace tokens from this candidate list with domain-specific tokens, expanding the vocabulary only when the replacement budget is exhausted. This hybrid replacement-then-expansion strategy enables us to prioritize recycling underutilized vocabulary slots, minimizing net parameter increase while maximizing domain vocabulary coverage.

Beyond standard benchmarking, we introduce a challenge-oriented evaluation framework that stress-tests model performance under conditions where domain vocabulary knowledge is critical.

We restructure the downstream domain-specific corpus to explicitly capture challenging scenarios: test sets with high out-of-vocabulary (OOV) concentrations in either source documents (SD) and reference summaries (RS)–OOV_SD and OOV_RS respec-tively. We also take a Random subset without any restriction on OOV concentration to compare the degree of performance in these challenging sce-narios. This targeted evaluation approach allows us to assess how well generalist models handle expert-level summarization tasks where domain-specific terminology is essential, providing a more rigorous test of vocabulary adaptation effectiveness beyond aggregate performance metrics. We evaluate our approach on two specialized domains—medical and legal literature—demonstrating that our method achieves competitive or superior performance compared to conventional vocabulary extension while substantially reducing parameter overhead and maintaining inference efficiency.

We hypothesize that the effectiveness of vocabulary adaptation is governed by the severity and location of lexical mismatch between pretrained tokenizers and downstream data. We find that: (i) across challenging scenarios of OOV_SD and OOV_RS, we observe more improvement in former setting over competing baselines. Although margins of gain in slightly higher in OOV_RS (4.44%) than OOV_SD (4.26%); (ii) performance gains are notably higher than the gains observed in Random setting (3.06%), validating that gains are higher in higher vocabulary mismatch scenario; (iii) vocabulary adaptation enables models to reach their best-performing checkpoints 35–55% earlier than continual pretraining alone, reducing the training time; (iv) hybrid replacement-then-expansion strategy remains highly parameter-efficient reducing parameters by 12.04% and 37.19% for Llama and Qwen models respectively averaged across both the domains. These results identify tokenization mismatch as a bottleneck in domain adaptation and motivate vocabulary-adaptation strategies as a tar-geted, data-dependent intervention. We make our codebase publicly available at https://github. com/gb-kgp/VocabReplace-Then-Expand.

Proposed Methodology (VOCABA DAPT) 2.1 Background Generalist LLMs are pretrained on broad-coverage corpora, resulting in tokenizers optimized for general text distributions. When deployed on specialized domains such as medical text, these tokeniz-ers exhibit systematic over-fragmentation. For in-stance, the term “Osteoporosis” is tokenized as [O, ste, opor, osis] by the Llama tokenizer, splitting into four subwords. This over-fragmentation introduces two primary challenges: first, the model must reconstruct semantic meaning across multiple token positions, increasing computational overhead and representation noise; second, generation becomes error-prone as the model must correctly predict each fragment in sequence, with errors compounding across token boundaries.

The standard solution to this vocabulary mismatch problem involves expanding the model’s vocabulary by adding domain-specific tokens. Let V src denote the source vocabulary of size |V src | with corresponding embedding matrix E ∈ R |V src |×d and unembedding matrix U ∈ R d×|V src | , where d represents the model’s hidden dimension. Adding k domain-specific tokens to form an expanded vocabulary V exp = V src ∪ V new necessitates expanding both embedding and unembedding matrices, introducing 2k · d additional parameters. For models with large hidden dimensions and substantial domain vocabularies, this parameter overhead becomes significant, increasing memory footprint and inference cost.

We propose an alternative approach that challenges the necessity of vocabulary expansion. Our central hypothesis is that generalist tokenizers contain a substantial subset of undertrained and unreachable tokens that contribute minimally to model performance. Rather than expanding the vocabulary, we identify these ineffectual tokens and replace them with domain-specific terminol-ogy, maintaining constant vocabulary size while addressing fragmentation. When domain requirements exceed the available candidate tokens, we resort to expansion only for the remaining terms, thereby minimizing parameter growth.

2.2 Identifying Candidate Tokens for Replacement

Our replacement strategy relies on identifying a candidate set V cand ⊆ V src comprising tokens that satisfy two independent criteria: they must be undertrained and unreachable.

The undertrained tokens are identified through the methodology of Land and Bartolo (2024) where the L2 norm for each token embedding e i in the vocabulary is computed, ∥e i ∥ 2 , excluding partial utf-8, fallback bytes, and unreachable tokens. Their

analysis demonstrates that tokens with embedding norms below a threshold corresponds to vocabulary items that appeared infrequently during pretraining and hence undertrained. This token token set is henceforth represented as V undertrained .

The unreachable tokens are identified through a consistency test (Land and Bartolo, 2024; Pura-son et al., 2026). A token t is deemed unreachable if decoding 1 its corresponding vocabulary token-id t i and encoding the decoded token does not yield the original token-id t i . E.g. decoding the encoding token-id 378 in Llama-3.1-8B results in âG, which upon encoding yield token-, id 5809. Formally, a token is unreachable when encode(decode(t i )) ≠ [t i ]. These tokens represent vocabulary entries that cannot be produced through the standard tokenization algorithm and thus remain inaccessible during normal model inference. While they occupy vocabulary slots and contribute to parameter count, they serve no functional role in model operation. This token set is henceforth represented as V unreachable .

We define our candidate set as the union of these two criteria:

Vcand = V undertrained ∪ V unreachable(1)

This union ensures we replace tokens that are poorly trained and inaccessible, providing a conservative strategy that minimizes risk of degrading model performance on general domains. Empiri-cally, we observe that approximately 3 − 4% per-cent of vocabulary tokens in both Llama-3.1-8B and Qwen2.5-7B satisfy the candidate set criterion, providing a substantial pool of replacement candi-dates.

We apply a final refinement to ensure tokenizer integrity. BPE (Byte-Pair Encoding) subword tok-enization algorithm construct vocabulary through iterative merge operations, where character sequences are progressively combined into larger units based on merge rules. Replacing a token that appears in the merge rule of another token outside the candidate set would fundamentally break the tokenization process, rendering certain vocabulary tokens untokenizable. To prevent this, we filter the candidate set to exclude any token that appears as a component in the merge rule of a token not designated for replacement. We construct a directed acyclic graph (DAG) with nodes as the token-id and an edge from token-i to token-j marking the relationship if token-i contributed in merge-rule of token-j (E.g., in → ing). Then, for every candidate that could be replaced, we checked if it has any descendants (nodes reachable from this node) that lies outside the candidate replacement set. If yes, we do not replace it, else we consider it for replace-ment. This set of tokens is marked as V exclude . This constraint guarantees that all remaining merge rules remain valid after vocabulary modification, preserving the deterministic and complete nature of the tokenization algorithm. The refined candidate set therefore contains only tokens that are undertrained, unreachable, and removing does not compromise the structural integrity of the tokenizer.

1 encoding and decoding here corresponds to buit-in tokenizer.encode and tokenizer.decode function calls of a model tokenizer.

V cand = V cand \V exclude


The final replacement candidate set is of size 1528 for Llama-3.1-8B (vocabulary size: 128K) and 3987 for Qwen-2.5-7B (vocabulary size: 151K). We next describe our domain-specific vocabulary construction step.

2.3 Building Domain-Specific Vocabulary

We construct domain-specific vocabulary through a process involving corpus curation, independent tok-enizer training, and vocabulary filtering for each target domain. This approach ensures that our added tokens genuinely represent domain-salient terminology rather than arbitrary subword fragments.

We curate two domain-specific corpora, each comprising 100 million tokens (100M) sampled from authoritative sources within their respective domains. The medical domain corpus is sampled from the MEDITRON pretraining corpora (Chen et al., 2023), which aggregates clinical practice guidelines, PubMed Central full-text articles, and article abstracts, providing comprehensive coverage of both clinical and biomedical language. For the legal domain, we compile a corpus from Supreme Court of India case documents, capturing the specialized vocabulary and linguistic conventions of Indian jurisprudence.

We train an independent Byte-Pair Encoding tokenizer using the HuggingFace tokenizers 2 library with a vocabulary size of 256,000 tokens dor each domain corpus. This training process learns domain-optimized merge operations that naturally surface frequently occurring domain-specific terms

as single tokens. From each trained domain tok-enizer vocabulary, we extract candidate tokens for addition to the base model. We filter this set to exclude any tokens that already exist in the source model vocabulary V src , as these tokens require no adaptation. This non-overlapping constraint ensures we only add genuinely new vocabulary items that address coverage gaps in the original tokenizer.

2 https://github.com/huggingface/tokenizers

We apply an additional refinement to ensure linguistic coherence across models and avoid introducing problematic tokens. We restrict the candidate set to tokens containing only English alphabetic characters, excluding any subwords that contain numeric digits, special symbols, or mixed alphanumeric patterns. This filtering serves multiple pur-poses: it eliminates formatting artifacts, date frag-ments, and identifier components that do not represent meaningful linguistic units; it ensures that added tokens correspond to genuine lexical items rather than incidental character sequences; and it maintains consistency with the predominantly alphabetic nature of established vocabulary in pre-trained models. The resulting filtered set forms D our domain-specific vocabulary V new , comprising high-frequency, domain-salient, purely alphabetic tokens that address the most significant tokeniza-tion inefficiencies for the target domain.

In both the settings, we select the top 10,000 vocabulary tokens ranked by frequency in the domain corpus, representing the most salient domain-specific vocabulary items. We next describe the procedure of vocabulary replacement.

2.4 Vocabulary Replacement-Then-Expansion and Embedding Initialization

D Thus far, we have a domain vocabulary V new and replacement candidate set V cand (Eq. 2), such that D |V new | > |V cand | . We first replace the V cand from LLM’s base vocabulary with equal sized set from D V new sorted by the natural merge order. We then expand the base vocabulary with the remaining D D |V new | − |V cand | elements from V new .

Initializing embeddings for the newly replaced and added tokens presents a critical challenge, as random initialization would require substantial training to achieve reasonable representations. Instead, we employ subword aggregation (Yam-aguchi et al., 2024), leveraging model’s existing understanding of subwords. For each new token t new , we tokenize it using the original tokenizer to obtain a sequence of source tokens [t 1 , . . . , t n ]. We then initialize the new token’s embedding as the mean of these constituent embeddings:

This initialization provides a reasonable starting point that captures compositional semantics while allowing subsequent training to refine the represen-tation. The same subword aggregation strategy is applied to initialize the corresponding unembed-ding matrix row. Next, we describe the procedure to tune the model with the modified vocabulary.

2.5 Domain-Specific Continual Pretraining

Following vocabulary modification, we conduct domain-specific continual pretraining to adapt the model to the target domain while training the new token representations. We employ Low-Rank Adaptation (LoRA) (Hu et al., 2022) to enable parameter-efficient training, inserting trainable low-rank matrices into the model’s attention and feed-forward layers while keeping the original pre-trained parameters frozen. This approach substantially reduces the number of trainable parameters and memory requirements during adaptation.

Each domain model is trained independently on a domain-specific corpus of 100M tokens sampled from high-quality sources representative as discussed previously. We train using the standard causal language modeling objective with next-token prediction, optimizing the model to predict each token given all preceding context. Training is conducted separately for medical and legal do-mains, producing two specialized model variants from each base model architecture.

3 实验设置
本节先介绍评估指标和所用数据集，再说明基线模型和实现细节。
数据集。 我们在两个领域的摘要数据集上测试本文方法。
医疗领域使用 MultiClinSumm 数据集（Lima López 等，2025）的英文子集。该数据集以临床病例报告作为源文档（SD），其对应的病例摘要作为参考摘要（RS）。法律领域使用 Shukla 等（2022）提出的抽取式摘要数据集（IN-ABS），其中 SD 是印度法院的判决书，RS 是该判决的抽象式摘要。为验证方法在不同任务上的泛化能力，医疗领域还补充了两项摘要任务：循证摘要（Mollá 和 Santiago-Martinez，2011）以及患者健康咨询摘要（Ben Abacha 和 Demner-Fushman，2019；Van Veen 等，2024）。EBM（循证摘要）任务的输入是一个查询及其对应的 PubMed 摘要（作为上下文／源文档），参考摘要则是针对该查询在该上下文下给出的答案。CHQ（患者健康咨询摘要）任务的输入是患者撰写的健康咨询问题，参考摘要是医学专家为该咨询写出的简洁单句问题。正文主要讨论临床报告摘要任务的结果，EBM 和 CHQ 数据集的结果见附录 A。
面向专家级摘要的数据集重构。 我们对标准数据集进行了重构，使得测试集由更具挑战性的数据点构成（Balde 等，2024、2025）。具体考虑两种场景：a) 源文档具有更高 OOV（词表外）集中度——记为 OOV_SD；b) 参考摘要具有更高 OOV 集中度——记为 OOV_RS。每类中 OOV 集中度最高的前 10% 数据点构成重构后的测试集，其余 90% 仍作为训练集。此外，我们还构建了一个等规模的 Random（随机）训练/测试子集，对 OOV 集中度不做任何限制，用以衡量在高难度场景下性能提升的程度。数据集统计信息见表 1。需要说明的是，两类挑战性场景的测试集之间存在大约 30%–40% 的重叠。
基线模型。 我们以两种 LLM 的基础版本——Qwen-2.5（Qwen 等，2025，模型 ID：Qwen/Qwen2.5-7B）和 Llama-3.1（Touvron 等，2023，模型 ID：meta-llama/Llama-3.1-8B）——作为 BASE 模型，它们未经过词表适配和持续预训练。此外，我们还使用了这两个基础模型在领域特定文本上进行持续预训练后的版本，记为"CPTOnly（无词表适配）"，用以单独评估词表适配带来的性能提升。
（表 1：法律和医疗领域在 Random、OOV_RS、OOV_SD 三种设置下的数据集统计，报告了平均 token 数、OOV 集中度（被切分超过一次的 unigram 占比）以及新词 unigram 集中度（RS 中未出现在 SD 中的 unigram 占比）。医疗领域的 OOV 集中度高于法律领域；法律领域的 token 数则远高于医疗领域。）
（表 2：受 ClinSumm（Van Veen 等，2024）启发设计的提示词结构。由于使用的是 BASE 模型，提示词中不区分系统提示和用户提示。）
训练与推理策略。 全部实验在单张 H100 80GB GPU 上完成。训练采用标准的下一词预测因果语言建模任务，推理使用贪心解码生成摘要。LoRA 的秩设为 32，alpha 设为 64，学习率为 2×10⁻⁵。在所有领域中，词表适配规模均为 1 万个 token，模型在 1 亿 token 的语料上训练 3 个 epoch，有效批大小为 64。CPTOnly 和 VOCABADAPT 使用完全相同的语料和超参数设置训练，VOCABADAPT 额外需要进行一次性的词表构建步骤，耗时约 30 分钟（在一台 Apple M3 Pro 笔记本的单核上完成）。尽管存在这部分额外开销，VOCABADAPT 的总训练耗时仍为 6.5–8.5 小时，明显快于 CPTOnly 所需的 10.5–12.5 小时。推理阶段采用上下文学习（in-context learning，Brown 等，2020），每个测试样本仅附加一个示例（关于 ICL 示例采样过程的细节见附录 A.1）。表 2 给出了 ICL 所用的提示词结构。
评估指标。 摘要质量评估以 Rouge-LCS（R-LCS）作为主要指标，报告其 F 值，遵循已有工作的做法（Balde 等，2024、2025；Fabbri 等，2021）。同时报告 BertScore（Zhang 等，2020），医疗领域使用 BioBert（Lee 等，2020）嵌入，法律领域使用 InLegalBERT（Paul 等，2023）嵌入。我们还对医疗和法律领域生成的摘要进行了 LLM-as-judge 评估：医疗领域使用谷歌的 MedGemma-27B 模型（Sellergren 等，2025），法律领域使用 Gemma3-27B 模型（Team 等，2025），从连贯性（coherence）、相关性（relevance）、忠实性（faithfulness）三个维度对模型生成的摘要进行 1–5 分评分（Fabbri 等，2021；Zhang 等，2023）。
4 实验结果
表 3 报告了 Rouge-LCS、BERTScore 以及碎片化分数（Fragment Score，即一个词被切分成的平均子词数），聚焦于最佳的词表适配策略。更多结果见附录 A。我们发现，词表扩展带来的影响与领域高度相关：在 OOV 集中度更高的医疗领域，性能提升更为显著；法律领域则相对有限。下面针对不同场景，详细讨论词表适配方法的有效与失效之处。
词表适配能降低碎片化分数。 词表适配技术能改善碎片化分数，从而减少过度碎片化、缓解词表不匹配问题。在医疗领域，挑战性 OOV 场景下 Llama 和 Qwen 的碎片化分数分别降低了 16.02% 和 15.63%；法律领域则分别降低 5.95% 和 5.73%。这种降低使模型在编码和生成时所需的 token 数更少，从而提升能效，并带来更好的表征。
词表适配在源文档 OOV 集中场景中的提升优于参考摘要 OOV 集中场景。 在 OOV_SD 场景下，词表适配在所有情形下（按 R-LCS 和 BERTScore 衡量）均优于 BASE，并在 8 项对比中 6 项优于 CPTOnly；而在 OOV_RS 场景下，词表适配在 8 项对比中 7 项优于 BASE，但仅 3 项优于 CPTOnly。这一现象可归因于源端 token 碎片化的降低幅度更大——OOV_SD 场景下降幅为 10.16%，而 OOV_RS 场景下仅为 8.92%。OOV_RS 场景中更高的碎片化程度会导致注意力分布更为分散，从而妨碍模型有效理解源文档，最终影响整体表现。
医疗领域的提升幅度高于法律领域。 尽管词表适配在两个领域都持续优于 BASE，但这一差异可以从表 1 的简单观察中得到解释：医疗领域的源文档和参考摘要均具有更高的 OOV 集中度，使其更适合词表适配方法发挥作用。
Random 设置下的提升幅度低于 OOV 设置。 在医疗领域，Random 设置下的绝对性能略低于挑战性 OOV 场景（Qwen：75.72 对比 OOV_SD 的 76.15 BSr；Llama：75.98 对比 OOV_SD 的 76.55 BSr），这验证了词表适配在 OOV 约束严重的场景下收益最大。Random 与 OOV 场景之间的性能差距在医疗领域更为明显，与该领域 SD 和 RS 子集中更高的 OOV 集中度相符。医疗领域中碎片化分数的降低幅度在 OOV 场景下也比 Random 场景更显著（Qwen：OOV 场景下 FrSr 为 1.10–1.14，Random 场景下为 1.09–1.10），说明 VOCABADAPT 性能的提升与碎片化降低幅度的提升是一致的。不过需要指出的是，即便在随机场景下，VOCABADAPT 仍带来了正向的（尽管较小的）影响。
词表适配能提升训练效率。 除最终性能外，词表适配方法在各个领域和模型系列中都能更早地达到最佳检查点，明显优于 CPTOnly。具体而言，从表 3 可以看出，词表适配方法相比 CPTOnly，达到最佳性能所需的训练步数减少了约 35%–55%，从而缩短了训练时间，同时保持甚至超越 CPTOnly 的性能表现。这表明，正确处理分词不匹配问题能够提升优化效率，使模型将更多能力分配给连贯的领域 token。
词表适配能提升语义重合度。 我们进行了一项简要分析，以理解为何某些情形下 Rouge-LCS 略有下降但 BERTScore 却有所提升。我们推测，词表适配会带来更强的抽象能力——即生成更多源文档中未出现的新 unigram。这类新词虽然在词面上与参考摘要重合度不高，但可能在语义上与之相近。相关结果见图 1：词表适配方法在各评估场景中都持续引入了更多有意义的抽象内容（新 unigram），相比基线方法更为明显，这也印证了"Rouge-LCS 略降而 BERTScore 上升"现象的成因。接下来需要回答的问题是：这些新词的引入是否提升了摘要的可读性和连贯性？我们通过 LLM-as-a-judge 评估来回答这一问题。
通过 LLM-as-a-Judge 评估摘要质量。 我们对 CPTOnly 和词表适配方法在医疗与法律领域生成的摘要进行了 LLM-as-a-Judge 评估（Croxford 等，2025），评估维度包括连贯性、相关性和忠实性，沿用已有研究的做法（Zhang 等，2023；Balde 等，2024、2025）。我们从医疗领域抽取 100 个随机样本，从法律领域抽取 20 个样本，均匀分布于各 OOV 场景和模型之间。平均得分见表 4。结果显示，相比有竞争力的 CPTOnly 基线，词表适配方法生成的摘要更具连贯性、相关性和忠实性（详见附录 A.4）。
（表 4：医疗领域使用 MedGemma-27B、法律领域使用 Gemma-27B 作为评判模型的 LLM-as-a-Judge 评估结果，涵盖连贯性、相关性、忠实性三个维度，评分范围 1–5。可见词表适配方法生成的摘要大多被评判模型打分高于 CPTOnly 基线，说明其摘要质量更优。）
（图 1：BASE、CPTOnly 和 VOCABADAPT 方法在 Llama 和 Qwen 模型生成摘要中的新 unigram 集中度中位数。可见词表适配方法相比基线引入了更多（有意义的）新词。）
有/无替换两种词表适配方法的消融分析。 表 5 报告了有替换和无替换两种词表适配方法的消融对比。结果显示，基于替换的策略在 16 项设置中的 13 项表现优于或与无替换策略持平。基于替换的策略对 Llama 的提升幅度（8 项设置中的 7 项）略高于对 Qwen 的提升幅度（8 项设置中的 6 项）。与此前关于医疗领域 OOV 集中度更高的讨论相反，本实验发现法律领域从替换策略中获益更多（全部 8 项设置），而医疗领域只有 8 项中的 5 项获益。一种可能的解释是法律领域的替换比例更高（25.43%），高于医疗领域（23.81%）。需要指出的是，基于替换的词表适配方法除了扩展的嵌入层和反嵌入层（lm_head）外，不会增加额外的可训练参数。我们注意到，基于替换的方法在 Llama-3.1 上节省了 12.04% 的参数，在 Qwen2.5-7B 上节省了 37.19% 的参数。
（表 5：有/无替换两种词表适配方法的消融分析。展示了词表规模、参数增量（百万）以及挑战性场景下的 R-LCS 和 BERTScore 指标。可见：(i) 基于替换的方法在 Llama-3.1 上节省 12.04% 参数、在 Qwen2.5-7B 上节省 37.19% 参数；(ii) 在 16 项设置中的 13 项中，替换方法表现优于无替换方法。）
与闭源 LLM 的对比。 我们对一个闭源模型 GPT-5（gpt-5-mini-2025-08-07）进行了零样本分析，旨在了解随着参数规模的增长，过度碎片化问题是否依然存在。为此，我们在 GPT-5 上运行了评估，并将结果与 Llama 和 Qwen 上 VOCABADAPT 方法的最佳结果进行了对比，结果见表 6。
我们的 7–8B 参数模型在结合词表适配后，在所有场景下均持续优于 gpt-5-mini（据推测其参数规模比 7B 模型大数个量级，且架构更复杂，可能包含 MoE 结构）。这一结果表明，即便对于更大规模的模型，词表适配依然具有应用价值。
（表 6：GPT-5-mini 与 VOCABADAPTBest 方法在医疗和法律领域摘要任务上的表现对比，使用 Rouge-LCS 和 BERTScore 在 OOV_SD、OOV_RS 挑战性场景及 Random 设置下进行评估。）



论文中提到的所有训练与评估相关参数汇总
一、模型与硬件
参数取值底座模型1Llama-3.1-8B（BASE版，model id: meta-llama/Llama-3.1-8B）底座模型2Qwen2.5-7B（BASE版，model id: Qwen/Qwen2.5-7B）Llama词表大小128,256（128K）Qwen词表大小151,665（151K）训练硬件单张 H100 80GB GPU
二、LoRA超参数
参数取值Rank32Alpha64Learning rate2×10⁻⁵训练目标标准因果语言建模（next-token prediction）Epoch数3有效Batch size64训练语料规模100M（1亿）token
三、词表适配相关参数
参数取值领域专用BPE分词器词表大小256,000（256K），医疗、法律各训练一个领域语料规模（构建词表用）各100M token新增/适配词表预算10,000（10K）个tokenLlama最终替换候选集大小1528Qwen最终替换候选集大小3987候选token占总词表比例约3-4%（未过滤前）词表构建耗时约30分钟（单核Apple M3 Pro笔记本）
四、训练耗时对比
方法训练总耗时VOCABADAPT6.5–8.5小时CPTOnly10.5–12.5小时训练时间节省幅度35–55%
五、最佳Checkpoint步数（Table 3 / Table 9）
任务/模型CPTOnly最佳步数VOCABADAPT最佳步数Medical Llama75003500Medical Qwen80003500Legal Llama100006500Legal Qwen105006500
六、参数增量与节省比例
模型/领域无替换参数增量有替换参数增量Llama医疗110M98MQwen医疗79M50MLlama法律91M79MQwen法律77M48M
整体节省比例数值Llama-3.1平均节省12.04%Qwen2.5-7B平均节省37.19%
替换比例数值法律领域替换比例25.43%医疗领域替换比例23.81%
七、数据集划分参数
参数取值OOV挑战集划分比例测试集取OOV集中度最高的前10%，剩余90%作训练集Random对照集等规模、无OOV限制的随机划分两类挑战集测试集重叠率约30–40%
八、推理/ICL参数
参数取值ICL示例数1个（one-shot）解码策略贪心解码（greedy decoding）医疗示例检索方法PubMedBERT句向量 + 余弦相似度法律示例检索方法BM25（用bm25s库实现）医疗摘要字数限制100词以内法律摘要字数限制300词以内CHQ任务摘要字数限制10词以内
九、LLM-as-a-Judge评估参数
参数取值医疗领域评判模型MedGemma-27B-text-it法律领域评判模型Gemma3-27B-it评估维度连贯性（coherence）、相关性（relevance）、忠实性（faithfulness）评分范围1–5评估方式三个维度分三次独立运行评分医疗领域抽样数量100个随机样本法律领域抽样数量20个随机样本
十、人工评估参数（附录A.5）
参数取值评估样本数20个摘要对（医疗领域）每个样本评估人数3名标注员独立评估总标注员人数12人评分维度流畅性（Fluency）、连贯性（Coherence）、相关性（Relevance）、事实一致性（Factual Consistency）评分范围1–5标注员时薪GBP 9/小时中位完成时间20分钟总成本约GBP 48（GBP 36标注费 + GBP 12平台费）标注平台Prolific
十一、评估指标体系
指标说明Rouge-LCS（R-LCS）基于最长公共子序列的F-scoreBERTScore（BSr）医疗用BioBert嵌入，法律用InLegalBERT嵌入Fragment Score（FrSrSD/FrSrRS）源文档/参考摘要的平均子词切分数Novel Unigram Concentration摘要中源文档未出现过的新词占比
