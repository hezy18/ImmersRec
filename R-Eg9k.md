Dear reviewer,

Thank you for your great effort! Here's the extra detail.

### W2-brief: MLP module. 
We substitute MLP with self-attention module for immersion prediction. Here shows the overall results of recommendation task (ImmersRec-att). **Bold** repesents the best performance of every backbone.

![image](https://github.com/hezy18/ImmersRec/assets/45138192/e93f3126-e20c-4a1e-8831-eddf5b15cd47) 


### W4-brief: time cost
The time for Step 1 (pretraining predictor on ImmersData) is 50 seconds per parameters group. Here is the time cost of Step 2. For each epoch, we training the ImmersRec and validate it. The validation is the cost of recommending the videos.

![image](https://github.com/hezy18/ImmersRec/assets/45138192/6dd7b876-d17b-4a32-9a3b-282dcee92bb0)

It is noted that for each sample, there are 1000 candidate items for validation. To maintain stable memory usage, we set the validation batch size to 8 (as the training batch size is 256). This does not affect the model performance but may increase the time required
