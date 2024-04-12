Dear reviewer,

Thank you for your great effort! Here's the extra detail.

 ## **Q1-brief**: Smaller K-values
 Here are the overall results. Five repeat experiments are conducted. The score shows the average performance and improvement. $^{*}$ and $^{**}$ indicates p-value < 0.05 and <0.1 from two-sided t-test, respectively. \textbf{bold} shows the higher result of the two settings. 
 ![image](https://github.com/hezy18/ImmersRec/figure/K_value_MicroVideo.png)

![image](https://github.com/hezy18/ImmersRec/figure/K_value_KuaiRand.png)

 ## **Q2-brief**: Time consumption 

The time for Step 1 (pretraining predictor on ImmersData) is 50 seconds per parameters group. Here is the time cost of Step 2. For each epoch, we training the ImmersRec and validate it. The validation is the cost of recommending the videos.

![image](https://github.com/hezy18/ImmersRec/figure/time_cost.png)


It is noted that for each sample, there are 1000 candidate items for validation. To maintain stable memory usage, we set the validation batch size to 8 (as the training batch size is 256). This does not affect the model performance but may increase the time required
