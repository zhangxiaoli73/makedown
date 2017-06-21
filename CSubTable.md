## CSubTable ##

**Scala:**
```scala
val model = CSubTable[Float]()
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

val model = CSubTable[Float]()

val input1 = Tensor[Float](5).rand()
val input2 = Tensor[Float](5).rand()
val input = T(input1, input2)
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
0.18143779
-0.24954873
-0.42380047
0.083815336
-0.10043772
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
Python Code
```
