# 运行该例子，可使用：
# ipython -i  example/1_ctgan_example.py
# 并查看 sampled_data 变量
from sdgx.statistics.single_table.ctabgan import CTABGAN
from sdgx.transform.sampler import DataSamplerCTGAN
from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.io.csv_utils import *

# 针对 csv 格式的小规模数据
# 目前我们以 df 作为输入的数据的格式
demo_data, discrete_cols = get_demo_single_table()


model = CTABGAN(epochs=10, transformer=DataTransformerCTGAN, sampler=DataSamplerCTGAN)
model.fit(demo_data, discrete_cols)

# sampled
sampled_data = model.sample(1000)
