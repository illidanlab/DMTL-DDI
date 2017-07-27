### Data description


-------------------------------------
#### [Original Data: ](https://drive.google.com/open?id=0BwRCD1KhUFkxSVFNNW5zZkNzRjg)

> FAERS DDI database: DDI_events.tsv

> Drug features: chemical_features.csv

-------------------------------------

#### [Integrating two datasets](https://drive.google.com/open?id=0BwRCD1KhUFkxVV9maXpIYVdYN3M)

> eventall_featlabel_clean_sym.npy

> eventall_featlabel_clean_sym.mat
```matlab
    data_dict = load('eventall_featlabel_clean_sym.mat');
    features = {data_dict.features};   % 125342 x 1232 features
    labels = {data_dict.labels};       % 125342 x 1318 labels
```
  Please refer to the **Data Description** section in the paper for the preprocessing and data description.

