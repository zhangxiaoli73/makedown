## SoftmaxWithCriterion ##

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

val input = Tensor[Float](1, 5, 2, 3).rand()
val target = Tensor(Storage((Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)
  
val normMode = NormMode.apply(2)
val model = SoftmaxWithCriterion[Float](normalizeMode = normMode)
val output = model.forward(input, target)
```

**Python example:**
```python
Python Code
```
