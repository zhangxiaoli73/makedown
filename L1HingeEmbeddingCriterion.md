## L1HingeEmbeddingCriterion ##

**Scala:**
```scala
val model = L1HingeEmbeddingCriterion[Float](0.6)
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

val model = L1HingeEmbeddingCriterion[Float](0.6)
val input1 = Tensor[Float](2).rand()
val input2 = Tensor[Float](2).rand()
val input = T(input1, input2)
val target = T(1.0f)

val output = model.forward(input, target)
```
output is
```
output: Float = 0.84714425
```

**Python example:**
```python
model = L1HingeEmbeddingCriterion(0.6)
input1 = np.random.randn(2)
input2 = np.random.randn(2)
input = [input1, input2]
target = [np.array([1.0])]

output = model.forward(input, target)
```
