# PRG-data-analysis
### feature_importance
![alt text](./readme_assets./compare_feature_importance_raw.png)
is this meaningful though?


### mlp
- 3 layers
- train_size = 12, test_size = 6
- train_batch_size = 3, test_batch_size = 3
- epochs = 3
![alt text](./readme_assets./output.png)

run with the following syntax:```dataset location```, ```input```, ```target``` <br>
```
python mlp.py "../sample_dataset/"  "lowlevel.csv" "mental_demand.npy"
```
