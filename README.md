# Road Network Guided Fine-Grained Urban Traffic Flow Inference
This work focus on how to accurately generate fine-grained data from coarse-grained data collected with a small number of traffic sensors, which is termed as fine-grained urban traffic flow inference.

## Requirements
Our RTFM uses the following dependencies: 

* Pytorch 1.5+
* Torchvision 0.6+
* CUDA 9.2 or latest version

## Framework
<!-- ![](img/framework.png) -->
![](imgs/RTFM_network.png)

## Results
We evaluate our method on TaxiBJ in four different time periods and the main experimental results are shown as follows:

![](img/results_BJ.png)

## Visualization
<!-- Here are the two areas in Beijing for which we provide dynamic visualizations. The first area is a place near Peking University, which is the same as in the Figure 9 of our paper. Second is a place near GuangQuMen Bridge, which is a main residential area in Beijing. 

| Area near Peking University | Area near GuangQuMen |
|-- |-- |
|![](img/gif/area0.png)|![](img/gif/area1.png)| -->

<!-- ### Area near Peking University
This is the visualization for the inferred distribution for area1 for a day, from 7:00 to 21:00. From the GIF below we can clearly see that when external factors are not considered (UrbanFM_ne), the inferred distribution remains very stable along with time changes. That is, there is no difference between flows in the evening and in the morning, which is intuitively inappropriate. However, the inference from UrbanFM is very dynamic and can adapt to time, which faithfully reflects that how people left from home to the research centers in the daytime and return home in the evening on weekdays, as well as different moving patterns on weekends.

| UrbanFM | UrbanFM_ne|
|-- |-- |
|![](img/gif/ext/area0/0_0.gif)|![](img/gif/ne/area0/0_0.gif)|
|![](img/gif/ext/area0/0_1.gif)|![](img/gif/ne/area0/0_1.gif)|

### Area near GuangQuMen
In this area, the top-right corner is the main crossroad where the other parts of this region are residences. When external factors are not considered, it can be seen that the model (UrbanFM_ne) only focuses on the crossroad and cannot adjust to different temporal factors. However, UrbanFM is free from this problem and produce adaptive flow inference. These visualizations suggest UrbanFM indeed considers the external factor for inference. 

| UrbanFM |  UrbanFM_ne|
|-- |-- |
|![](img/gif/ext/area1/0_0.gif)|![](img/gif/ne/area1/0_0.gif)|
|![](img/gif/ext/area1/0_1.gif)|![](img/gif/ne/area1/0_1.gif)| -->




<!-- If you find this code and dataset useful for your research, please cite our paper:

```
``` -->

## Data preparation
The datasets *XiAn* and *ChengDu* we construt is detailed in Section 4.1.1 of our paper. Here, we release them for public use. The corresponding external factors data (e.g., meteorology, time features) are also included. 

User can simply unzip "./P1.zip" to a folder named P1 to obtain the training and test data. For example, the path of training input need to be "./data/P1/train/X.npy".
'''
path/data/XiAn/train/
                    X.npy/    # coarse-grained traffic flow maps
                    Y.npy/    # fine-grained traffic flow maps
                    ext.npy/  # external factor vectors
'''


## Model Training
Examples of input arguments:
- channels: input and output channel (2 for XiAn and ChengDu)
- ext_flag: whether to use external factor fusion subnet
- dataset: which dataset to use

The following examples are conducted on dataset XiAn:
* Example 1 (default settings):
```
python -m RTFM.train --ext_flag --dataset "XiAn"
```

* Example 2 (using arbitrary settings):
```
python -m RTFM.train --ext_flag --n_epochs 200 --n_residuals 20 --base_channels 128 --dataset "XiAn"
```

* Example 3 (UrbanFM-ne, i.e., UrbanFM without external subnet):
```
python -m UrbanFM.train --dataset "P1"
```

<!-- * Example 4 (UrbanFM with large amounts of parameters):
```
python -m UrbanFM.train --ext_flag --n_residuals 16 --base_channels 128 --dataset "P1"
``` -->


## Model Test
To test above trained models, you can use the following command to run our code:
```
python -m UrbanFM.test --ext_flag --n_epochs 200 --n_residuals 20 --base_channels 64 --dataset "P1"
```

## License
UrbanFM is released under the MIT License (refer to the LICENSE file for details).
