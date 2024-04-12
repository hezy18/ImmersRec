Dear reviewer,

Thank you for your great effort! Here's the extra detail.

### W2-brief: Necessity

Here is the statistcs of like in the datasets.

| **null** | **KuaiRand:** | **MicroVideo** |
|----------|---------------|----------------|
| #view:   | 4,192,604     | 1,365,564      | 
| #like    | 61,363        | 26,111         |

We conducted  studies where training on view and evaluated by likes. The performance is here: 

| **backbons** | **settings** | **HR@20** | **NDCG@20** | **HR@50** | **NDCG@50** |
|--------------|--------------|-----------|-------------|-----------|-------------|
| FinalMLP     | baseline     | 0.2825    | 0.1401      | 0.4579    | 0.1831      |
|              | ImmersRec    | 0.3321    | 0.1643      | 0.4914    | 0.2047      |
| WideDeep     | baseline     | 0.3683    | 0.1658      | 0.5098    | 0.1940      |
|              | ImmersRec    | 0.4440    | 0.2299      | 0.5814    | 0.2667      | 


### W3-brief: Dataset Small

Here is citations of the works using KuaiRand-1k.

* Liu, Ziru, et al. "Multi-task recommendations with reinforcement learning." WWW 2023.
* Wang, Yejing, et al. "Single-shot Feature Selection for Multi-task Recommendations." SIGIR 2023.
* Zhang, Qing, et al. "Debiasing recommendation by learning identifiable latent confounders." KDD 2023.
