#**论文框架：**
作者提出一种基于多任务的网络串联方法（Multi-task Network Cascade），解决问题实例分割（Instance-aware Semantic Segmentation）。该模型分成三个子网络，differentiate instance，estimate mask，category object。分别针对三个问题，获取region-level的检测框，得到pixel-level的mask，对每个mask获得category-level的label。

流程图如下：
![Multi-task Network Cascade](http://img.blog.csdn.net/20170807101308993?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRXRoYW5fV3V1dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中conv feature map使用VGG-16获得。
并且，该模型不同于一般的多任务网络，他的后一任务的loss依赖于前一任务的输出，所以他的三个loss都是不独立的。

**I.  Regressing Box-level Instance**
第一个子网络获得每个目标的bounding box，和objectness score（class-agnostic，即不知道该目标属于哪一类）。
这个子网络使用Region-Proposal-Networks（RPN）的网络框架，输入是shared feature map（即特征图），连接一个3X3的卷积层用来降维，再接一个1X1的卷积层来回归box location和classifying object/non-object。
作者采用RPN的loss function ：
$$L_1=L_1(B(\Theta))$$
其中$\Theta$表示所有待优化的网络参数。$B$ 表示该网络，输出是一系列的boxes：$B=\{B_i\}$ ,$B_i=\{x_i,y_i,w_i,h_i,p_i\}$ ,$i$ 表示每个box的索引，$(x_i,y_i)$ 表示中心位置坐标，$(w_i,h_i)$ 表示长宽，$p_i$ 表示objectness probability。

**II. Regressing Mask-level Instance**
在stage-2，输入：shared features 和 stage-1 box，输出：基于每个box的pixel-level segmentation mask。在这部分，mask-level instance 依旧是class-agnostic。
对于输入的stage-1 box，使用Region-of-interest（RoI） pooling 提取box特征（目的是从feature map上获取对应的任意尺寸的box的特征）。再接两个全连接层（fc layer），前一个负责降维至256，后一个fc layer负责回归pixel-level mask。
$Loss$ 函数：
$$L_2=L_2(M(\Theta)|B(|\Theta))$$
其中$M$ 表示第二个子网络，输出一系列mask：$M=\{M_i\}$ ,其中$M_i$ 是一个$m^2$ 维的逻辑回归输出（属于[0,1]二值化，$m^2$ 是每个mask的分辨率）。
可以看出，$L_2$ 同时受限于$M$ 和$B$ 。

**III. Category Instance**
stage-3，输入：shared features，stage-1 box和stage-2 mask。输出：每个Instance的category score。
在stage-2中，我们提取了每个box的feature，现在我们继续提取每个mask的feature：
$$F_i^{Mask}(\Theta)=F_i^{RoI}(\Theta)*M_i(\Theta)$$
其中$F_i^{RoI}(\Theta)$ 表示经过RoI pooling 提取的box的feature。
然后作者考虑同时使用masked feature 和 box-based feature。通过连接softmax classifier 预测N+1种类。
Loss函数如下：
$$L_3=L_3(C(\Theta)|M(\Theta),B(\Theta))$$

#**技术挑战：**
**End-to-end Training**
本文定义了一个整体的Loss function：
$$L(\Theta)=L_1(B(\Theta))+L_2(M(\Theta)|B(|\Theta))+L_3(C(\Theta)|M(\Theta),B(\Theta))$$
但不同于普通的多任务学习，该论文的后一级任务都是基于前一级的输出。

在end-to-end训练任务中，最主要的技术挑战就是在使用RoI pooling的过程中，预测框$B_i(\Theta)$ 的空间位置不断变化。因为在fast R-CNN论文中使用的RoI pooling 的预测框是经过预训练的，它的反向传播只考虑$F(\Theta)$ ,但我们现在必须同时考虑$B_i(\Theta)$ 。

**Differentiable RoI Warping Layers**
原始的RoI pooling layer 是在一个box内的离散网格执行max pooling。所以本文要在RoI pooling layer中，增加$B_i(\Theta)$ 信息，以达到本文的需求。
大体的计算公式如下：
$$F_i^{RoI}(\Theta)=G(B(\Theta))F(\Theta)$$
$F(\Theta)$ 就是feature map。G表示裁剪和变形操作的矩阵，就是任意输入一个$B_i(\Theta)$， 都将它裁剪变形到固定大小，然后变形成一个行列式矩阵，目的就是将$n$ 维的$F(\Theta)$ 转换成$n'$ 维的$F_i^{RoI}(\Theta)$。 
具体实现略（之后补充）。

**Masking layers**
在$L_3$ 公式中，$L_3$ 的计算也依赖于$M(\Theta)$ , $B(\Theta)$ 。我们经过了Differentiable RoI Warping Module，就可以将$L_3$ 公式简单通过一个element-wise product module进行计算。

剩余略（之后补充）。
