# CS229-note1-学习心得

## **第一部分**

### 线性回归

#### 简介：

##### 1.背景：地区 *房间大小和价格*  数据，推进如何执行监督学习，以及确定参数 *thetaθ* 。

将 **y **假设为关于 **x** 的线性函数，**θ** 代表拟合参数

于是：

![image-20230416003859010](C:\Users\hui\AppData\Roaming\Typora\typora-user-images\image-20230416003859010.png)

为了简单起见，假定 **X0 = 1**

于是：

![image-20230416004031857](C:\Users\hui\AppData\Roaming\Typora\typora-user-images\image-20230416004031857.png)

**如何确定 *θ*** ？

​	于是：

![image-20230416004300824](C:\Users\hui\AppData\Roaming\Typora\typora-user-images\image-20230416004300824.png)

​																										注：1/2参考方差

------



##### **2.LMS算法-最小均方，也称为 Widrow-Hoff 学习规则**，也即最小梯度下降

为了使J(θ) 更⼩，直到我们希望收敛到使 J(θ) **最⼩化**的 θ 值

于是：

![image-20230416110041834](../../../AppData/Roaming/Typora/typora-user-images/image-20230416110041834.png)

​					注：α 称为学习率。这是⼀种⾮常⾃然的算法，它会在 J 的最陡下降⽅向上重复迈出⼀步

​																						**公式2-1**

假设我们有一个训练实例（x,y）,我们需要计算出右侧的偏导数

于是：

![image-20230416110230249](../../../AppData/Roaming/Typora/typora-user-images/image-20230416110230249.png)

​																						**公式 2-2 **

接着，结合**公式2-1和2公式-2**

于是：

![image-20230416110653930](../../../AppData/Roaming/Typora/typora-user-images/image-20230416110653930.png)

​						  	**注：LMS更新规则（LMS代表“最小均方”），也称为 Widrow-Hoff 学习规则。**



如果在拟合过程中，我们的预测值和实际值有较大出入，以下是两种解决方法：

***First*.对于包含多个训练集，替换为以下算法，重复直到收敛：**

![image-20230416111524721](../../../AppData/Roaming/Typora/typora-user-images/image-20230416111524721.png)

​												注：求和部分为   ***∂J(θ)/∂θj   (for the original definition of J).***

* 此方法被称为**批梯度下降**，每一步都扫描整个**训练集**的每个实例
* 此方法受**局部极小值**影响大，但是**线性回归问题**只有一个全局最优值，因此此梯度下降总是**收敛**
* 假设 **学习率 α**  不太大

------

**下面是对比两种方法的直观图：**

![image-20230416113007023](../../../AppData/Roaming/Typora/typora-user-images/image-20230416113007023.png)

​																						**梯度下降实例**

![image-20230416113503115](../../../AppData/Roaming/Typora/typora-user-images/image-20230416113503115.png)

​																				**批梯度下降实例**

***Second*.考虑以下算法：**

![image-20230416113826870](../../../AppData/Roaming/Typora/typora-user-images/image-20230416113826870.png)

* 根据**单个训练样例**相关误差梯度更新参数
* **随机梯度下降**
* 不用每次都扫描**整个训练集**
* 更快 ***接近***  **θ最小值**，可能永远无法到达最小值，在**最小值附近振荡**

------



##### 3.正规方程

不依靠**迭代算法**求最小值的算法

* **学习率 α** 减小到 0 
* **参数 θ** 收敛到最小值而**非振荡**

###### 3.1矩阵导数

为了简化书写以及，避免大量的导数矩阵出现

于是：

**函数 f** 对于 **矩阵 A** 的导数：

![image-20230416203613560](../../../AppData/Roaming/Typora/typora-user-images/image-20230416203613560.png)

​																											**公式1 **

我们定义 **矩阵A：**												![image-20230416203928430](../../../AppData/Roaming/Typora/typora-user-images/image-20230416203928430.png)

**函数 f ：**

![image-20230416203859510](../../../AppData/Roaming/Typora/typora-user-images/image-20230416203859510.png)

​																												**公式2**



因此由**公式1和2**得出：

![image-20230416204134728](../../../AppData/Roaming/Typora/typora-user-images/image-20230416204134728.png)

如果 矩阵A 和 矩阵B相乘，使得矩阵AB为方阵，那么有：**tr*AB*  =  tr*BA***

以下再简单介绍几条性质，**（4）仅适用于非奇异方阵**

![image-20230416205506467](../../../AppData/Roaming/Typora/typora-user-images/image-20230416205506467.png)

**（1）阐述：**
$$
B ∈ R
n×m.
$$

$$
f : R
m×n
→ R
$$

$$
f(A) = trAB.
$$

$$
A ∈ R
m×n

$$

矩阵A，B相乘为**方阵**，于是在矩阵B中，N行都被赋予矩阵A的M_i。类似的，列元素也是，对函数求导后，A的每一项求导为1，即**m行，n列被保留**。即为**B的转置**。

**（2）阐述：**根据（1），想象一下，如果求导对象是A的转置呢？

**（3）阐述：**根据（1），（2）。

###### 3.2最小二乘再访

给定一个矩阵 X 为 m * n

把要训练的实例输入X的行中：

![image-20230416212526442](../../../AppData/Roaming/Typora/typora-user-images/image-20230416212526442.png)

目标值定义为 Y 向量：

![image-20230416212811562](../../../AppData/Roaming/Typora/typora-user-images/image-20230416212811562.png)

于是：
$$
hθ(x
(i)
) = (x
(i)
)
T
θ
$$
所以：

![image-20230416212919626](../../../AppData/Roaming/Typora/typora-user-images/image-20230416212919626.png)

根据<img src="../../../AppData/Roaming/Typora/typora-user-images/image-20230416213240888.png" alt="image-20230416213240888" style="zoom:50%;" />

我们有：

![image-20230416213040819](../../../AppData/Roaming/Typora/typora-user-images/image-20230416213040819.png)



结合（2），（3）我们得出：

![image-20230416213348942](../../../AppData/Roaming/Typora/typora-user-images/image-20230416213348942.png)



推出：

<img src="../../../AppData/Roaming/Typora/typora-user-images/image-20230416213423818.png" alt="image-20230416213423818" style="zoom:80%;" />

**注：**

* 推导步骤3，**实数的迹就是实数**
* 推导步骤4，**矩阵A的迹就是矩阵A转置的迹**
* 推导步骤5，使用上述等式（5）
* 其中 **AT = θ, B = BT = XT X, and C = I**
* 然后使用了上述等式（1）

接下来将 **J(θ)** 的导数设为 **0**，得到：

​														<img src="../../../AppData/Roaming/Typora/typora-user-images/image-20230416214001538.png" alt="image-20230416214001538" style="zoom: 67%;" /><img src="../../../AppData/Roaming/Typora/typora-user-images/image-20230416214018377.png" alt="image-20230416214018377" style="zoom:67%;" />

##### 4.概率解释

##### 5.局部加权线性回归

## 第二部分

### 分类和逻辑回归