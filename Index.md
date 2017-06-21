## Index ##

**Scala:**
```scala
val model = Index[Float](1)
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
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor[Float](3).rand()
val input2 = Tensor[Float](4)
input2(Array(1)) = 1
input2(Array(2)) = 2
input2(Array(3)) = 2
input2(Array(4)) = 3

val input = T(input1, input2)
val model = Index[Float](1)
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
0.6175723
0.4498806
0.4498806
0.41750473
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
```

**Python example:**
```python
Python Code
```
