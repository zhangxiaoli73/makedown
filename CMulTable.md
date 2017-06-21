## CMulTable ##

**Scala:**
```scala
val model = CMulTable[Float]()
```
**Python:**
```python
model = CMulTable()
```

Description

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val model = CMulTable[Float]()

val input1 = Tensor[Float](5).rand()
val input2 = Tensor[Float](5).rand()
val input = T(input1, input2)
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
0.45114923
0.04661638
0.31990767
0.24304615
0.56568104
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
model = CMulTable()
t1 = np.random.randn(5)
t2 = np.random.randn(5)
input = [t1, t2]
model.forward(input)
```
output is
```
array([ 3.83480358, -1.08372796,  0.08163583, -1.92096496,  0.39564416], dtype=float32)
```
