## Clamp ##

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

val model = Clamp[Float](-10, 10)
val input = Tensor[Float](2, 2, 2).rand()
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.19594982	0.1558478	
0.23255411	0.8538258	

(2,.,.) =
0.76815903	0.0132634975	
0.33081427	0.5836359	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]
```

**Python example:**
```python
Python Code
```
