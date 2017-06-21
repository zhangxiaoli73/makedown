## SoftmaxWithCriterion ##

**Scala:**
```scala
val model = SoftmaxWithCriterion[Float](normalizeMode = normMode)
```
**Python:**
```python
Python cod, how to new an instance
```

Description

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

val input = Tensor[Float](1, 5, 2, 3).rand()
val target = Tensor(Storage(Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)
  
val normMode = NormMode.apply(2)
val model = SoftmaxWithCriterion[Float](normalizeMode = normMode)
val output = model.forward(input, target)
```
output is
```
output: Float = 10.494613
```
**Python example:**
```python
Python Code
```
