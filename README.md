## This is a pytorch implementation of the paper *[Zhen Zhang*, Yuewi Ming, Quming Shen, Yanyu Wang, Yuhui Zhang.  An extended variational autoencoder for cross-subject electromyograph gesture recognition. Biomedical Signal Processing and Control. 2025, 99, 106828]*，请随便学习使用，如有发表文章请引用该文章

#### Environment
- Pytorch 
- Python 

#### files Structure

```
--paper_experiments
        | --data
                  |--dataset: 肌电数据
        | --vae_semg
                  |--dataset:
 			  |--semgdata_loader.py:数据加载
                  |--supervised--
			  |--data_process.py:数据处理函数
                          |--model_XXX.py:模型文件
                          |--experiment_XXX.py:实验主程序，可直接执行
                          |--evalution_XXX.py: 评估程序，需要中间保存的数据


