## SoftmaxWithCriterion ##

**Scala:**
```scala
val model = SoftmaxWithCriterion[Float]()
```
**Python:**
```python
model = SoftmaxWithCriterion()
```

Description

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

val input = Tensor[Float](1, 5, 2, 3).rand()
val target = Tensor(Storage(Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)

val model = SoftmaxWithCriterion[Float]()
val output = model.forward(input, target)
```
output is
```
output: Float = 10.494613
```
**Python example:**
```python
input = np.random.randn(1, 5, 2, 3)
target = np.array([[[[2.0, 4.0, 2.0], [4.0, 1.0, 2.0]]]])

model = SoftmaxWithCriterion()
output = model.forward(input, target)
```
output is
```
2.1241186
```
