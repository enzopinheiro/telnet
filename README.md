# TelNet
[Click here to read the paper](https://www.nature.com/articles/s43247-025-02207-2)

This project contains the codes necessary to reproduce the results of "An interpretable machine learning model for seasonal precipitation forecasting" by E. Pinheiro and T. B. M. J. Ouarda.
It requires about 16GB of VRAM and takes about a 4 days to run the full sampling procedure on a single A100 GPU. 
Using different versions of the libraries may lead to different results due to stochastic nature of the algorithms, thus it is recommended to use the container provided. Further instructions on how to mount and use the container are provided in docker directory.
The dataset used in the paper can be downloaded from the following link: 
https://drive.google.com/file/d/1ZZvyF-nRT_k7aqFCDpOBCMt0_UenRT-W/view?usp=sharing
Once the dataset is downloaded, an environmental variable "TELNET_DATADIR" should be set to the path of the dataset.

