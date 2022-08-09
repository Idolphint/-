目标是：判断企业贷款违约概率

可用的数据：企业所属行业数据，四级，用个四维向量能代表or用一个向量就能代表？



## 数据处理

行业基本信息，可以用的基本只有行业代码、纳税申报信息+交税信息、投资人信息。

目标是分类

行业代码直接embedding.

纳税和交税对齐之后构成无顺序数组，然后按照时间排序，序列化处理

投资人embedding，投资比例和金额？。

## 模型

首先确定行业代码个数和投资人个数，投资人个数太多了，没有嵌入的必要

行业代码个数为486，需要压缩一下成为10维的？



全拿来筛，得到的是：

fea 7，16，23，59，86， 88， 163，174，176，187，225

ret_non0_month,  month20_pay, stage_2_ret_rate, stage_0_ret_rate, stage_5_ret_rate



0，11，16, 38，55，77，88，99，132，165，176，

ret_non0_month， month2_ret,  month6_ret, month1_pay, stage_2_ret_rate, stage_0_pay_rate, stage_1_ret_rate,  stage_3_pay_rate