## SoftPlus ##

**Scala:**
```scala
Scala code, how to new an instance
```
**Python:**
```python
Python cod, how to new an instance
```

Description

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val layer = SoftPlus[Float]()
val input = Tensor[Float](2, 3, 4).rand()
val output = layer.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.7835901	0.9962422	0.7521913	1.1398541	
0.71477914	1.0578016	0.7928455	0.6975596	
0.8969599	0.88751	0.7120005	0.9749961	

(2,.,.) =
1.0419492	0.76816565	1.0519257	1.1393704	
0.72922856	1.1668311	0.7839851	0.94722736	
0.8995574	1.1064267	1.1123171	0.92133987	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
**Python example:**
```python
Python Code
```
