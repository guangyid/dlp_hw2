## 网络结构

### Lenet

| layers | name              | 操作                                                             | 输出                                              |
| ------ | ----------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| 1      | input层(**输入层**)   | 归一化                                                            | 32\*32的图像(本层不算LeNet-5的网络结构，传统上，不将输入层视为网络层次结构之一) |
| 2      | C1层(**卷积层**)      | 6个5\*5卷积核                                                      | feature map：28\*28\@6                           |
| 3      | S2层(**池化层**、下采样层) | 6个2\*2的pooling(对C1中的2\*2区域内的像素求和乘以一个权值系数再加上一个偏置，然后将这个结果再做一次映射) | 14\*14\@16                                      |
| 4      | C3层(**卷积层**)      | 16个5\*5卷积核                                                     | 10\*10\@16                                      |
| 5      | S4层(**池化层**)      | 16个2\*2的pooling                                                | 5\*5\@16                                        |
| 6      | C5层(**卷积层**)      | 120个5\*5卷积核                                                    | 1\*1\@120                                       |
| 7      | F6层(**全连接层**)     | 加权偏置，`$sigmoid$`激活                                             | 84个结点(7\*12的比特图)                                |
| 8      | output层(**全连接层**) | RBF:`$y_i=\sum_{j}(x_j-w_{ij})^2,0<i<9,0<j<7*12-1$`            | 10个结点                                           |

### Alexnet

AlexNet整体的网络结构包括：1个输入层（input layer）、5个卷积层（C1、C2、C3、C4、C5）、2个全连接层（FC6、FC7）和1个输出层（output layer）。

| layers | name | 操作                              | 输出 |
| ------ | ---- | ------------------------------- | -- |
| Input  | 输入层  |                                 |    |
| C1、C2  | 卷积层  | 卷积-->ReLU-->局部响应归一化（LRN）-->重叠池化 |    |
| C3、C4  | 卷积层  | 卷积-->ReLU                       |    |
| C5     | 卷积层  | 卷积-->ReLU-->重叠池化                |    |
| FC7    | 全连接层 | 全连接 -->ReLU -->Dropout          |    |
| output | 输出层  | 全连接 -->Softmax                  |    |

[torch代码](https://blog.csdn.net/qq_45195178/article/details/127660141)

![image](https://img-blog.csdn.net/20180116220229382?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbG92ZWxpdXp6/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

Alex net的torch源码和论文中的通道变化有些出入，应该是但是作者用了两块gpu引起的

### resnet

[源码](https://blog.csdn.net/u014453898/article/details/97115891)

![image](https://img-blog.csdnimg.cn/2019072411484875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ0NTM4OTg=,size_16,color_FFFFFF,t_70 "image")
