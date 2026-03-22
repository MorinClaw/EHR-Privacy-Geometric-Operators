# 论文小节：Privacy Evaluation Protocol

以下可直接作为论文中「Privacy Evaluation and Attack Suite」一节的写作提纲与实验矩阵说明。

---

## 1. Threat Models and Leakage Regimes

We evaluate the numeric transformation operators under the following assumptions (Kerckhoffs principle):

- The adversary **knows** the operator family and implementation (T1/T2/T3, α, code).  
- The adversary **does not** hold a long-term key and **cannot** control the per-record random seed.  
- The adversary may have **additional side information** depending on the leakage level.

**Leakage levels** (aligned with the 20% / 0.01% / no-pairs settings in prior experiments):

| Symbol | Name                 | Description                                      |
|--------|----------------------|--------------------------------------------------|
| **L0** | No-pairs             | Adversary sees only many \(y = T(x)\), no \((x,y)\) pairs. |
| **L1** | Few-pairs            | Very few pairs, e.g. 0.01% of \((x,y)\).        |
| **L2** | Many-pairs           | Larger fraction, e.g. 20% of \((x,y)\).         |
| **L3** | Full-pairs (oracle)  | Almost all \((x,y)\); upper bound on invertibility. |

Optional side information (S1: same-distribution raw data; S2: downstream model on \(y\)) can be deferred to appendix or future work.

---

## 2. Attack Families

We organize the evaluation into four attack families.

### A. Pointwise Reconstruction

- **Game:** Given \(y = T(x)\), the adversary outputs \(\hat{x} = A(y)\) to approximate \(x\).  
- **Attackers:** Linear/Ridge, MLP (and optionally CNN/Transformer); we report the **best** over the family.  
- **Leakage:** Under L0, the adversary has no pairs (baseline: constant or \(\hat{x}=y\)); under L1/L2/L3, the adversary trains on the corresponding fraction of \((y,x)\) pairs.  
- **Metrics:** \(R^2\), Hit-rate within \(\delta\sigma\) for \(\delta \in \{0.1, 0.5, 1.0\}\), MAE in z-space, Pearson/Spearman \(\mathrm{corr}(\hat{x}, x)\).

### B. Record Linkage / Re-identification

- **Game:** The adversary is given a candidate set \(\{x^{(1)},\ldots,x^{(m)}\}\) containing the true \(x^*\), and receives \(y = T(x^*)\). He outputs an index \(\hat{\jmath}\); success if \(x^{(\hat{\jmath})} = x^*\).  
- **Metrics:** Re-id@1, Re-id@k (e.g. \(k=5\)), and AUC; compare to random baseline \(1/m\).

### C. Membership Inference

- **Game:** The adversary must decide whether a candidate raw record \(x\) belongs to the exported (training) cohort.  
- **Data-level MI:** Use summary features of \(y\) (and optionally \(x\)) to train a binary classifier (member vs non-member).  
- **Metrics:** ROC-AUC and Advantage \(\mathrm{AUC} - 0.5\).

### D. Attribute Inference and Indistinguishability

- **Attribute inference:** Define sensitive functions \(f(x)\) (e.g. \(\max_t x_t\), value at \(t=24\), or “ever above 90th percentile”). Train a predictor from \(y\) to \(f(x)\); report \(R^2\) or AUC as appropriate.  
- **Pairwise indistinguishability (IND-style):** Sample \((x_0, x_1)\) and bit \(b\); give \(y = T(x_b)\). The adversary guesses \(b\). Report \(\Pr[\hat{b}=b]\) and advantage \(|\Pr[\hat{b}=b]-0.5|\).

---

## 3. Metrics (Formulas)

- **Reconstruction:**  
  \(R^2 = 1 - \frac{\sum_{i,t}(x_{i,t}-\hat{x}_{i,t})^2}{\sum_{i,t}(x_{i,t}-\bar{x})^2}\);  
  \(\mathrm{Hit}_\delta = \Pr\bigl(|\tilde{\hat{x}}_{i,t} - \tilde{x}_{i,t}| \le \delta\bigr)\) in z-space;  
  \(\mathrm{MAE}_z = \mathbb{E}|\tilde{\hat{x}} - \tilde{x}|\).

- **Record linkage:** Re-id@1 = \(\Pr[\hat{\jmath} = \text{true index}]\); Re-id@k = \(\Pr[\text{true in top-}k]\); AUC for same/different pair classification.

- **Membership:** AUC of member vs non-member classifier; Advantage = \(\mathrm{AUC}-0.5\).

- **Distinguishability:** Accuracy of guessing \(b\) from \((x_0,x_1,y)\); Advantage = \(|\Pr[\hat{b}=b]-0.5|\).

---

## 4. Experiment Matrix

- **Data:** Fixed cohort; Train/Val/Test split by **stay** (e.g. 60/20/20). All attacks are trained on Train, tuned on Val, and evaluated on Test.  
- **Operators:** \(T \in \{\mathrm{T1},\mathrm{T2},\mathrm{T3}\}\).  
- **α:** \(\alpha \in \{0.3, 0.5, 0.8, 1.0\}\) (must match the precomputed perturbed data).  
- **Leakage:** L0, L1, L2, L3 for Attack A; fixed settings for B/C/D as described above.  

Table (example): for each (T, α, leakage), report best-attack \(R^2\), Hit-rates, Re-id@1, AUC (MI), and distinguishability advantage.

---

## 5. Relation to Cryptographic and DP Notions

- **Reconstruction / attribute inference** correspond to violating *semantic security* (recovering plaintext or a function of it).  
- **Record linkage** corresponds to *re-identification / de-anonymization*.  
- **Membership inference** corresponds to the standard *DP membership game*.  
- **Pairwise indistinguishability** corresponds to an *IND-CPA–style distinguishing game* at the record level.

This framing makes the evaluation interpretable for reviewers familiar with cryptography and differential privacy.
