Dear reviewer,

Thank you for your great effort! Here's the extra detail.

 ## **Q1-brief**: Smaller K-values
 Here are the overall results. Five repeat experiments are conducted. The score shows the average performance and improvement. $^{*}$ and $^{**}$ indicates p-value < 0.05 and <0.1 from two-sided t-test, respectively. \textbf{bold} shows the higher result of the two settings. 
 ![image](https://github.com/hezy18/ImmersRec/assets/45138192/caf17a6b-8be0-4395-80a0-5f0ee05f2661)

![image](https://github.com/hezy18/ImmersRec/assets/45138192/1a3be2fb-52aa-43a8-8018-5d8bb4071166)

 ## **Q2-brief**: Time consumption 

The time for Step 1 (pretraining predictor on ImmersData) is 50 seconds per parameters group. Here is the time cost of Step 2. For each epoch, we training the ImmersRec and validate it. The validation is the cost of recommending the videos.

![image](https://github.com/hezy18/ImmersRec/assets/45138192/6dd7b876-d17b-4a32-9a3b-282dcee92bb0)


It is noted that for each sample, there are 1000 candidate items for validation. To maintain stable memory usage, we set the validation batch size to 8 (as the training batch size is 256). This does not affect the model performance but may increase the time required
