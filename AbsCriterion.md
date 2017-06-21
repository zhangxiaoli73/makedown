## AbsCriterion ##

**Scala:**
```scala
val criterion = AbsCriterion[Float]()
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
val criterion = AbsCriterion[Float]()

val input = Tensor[Float](3).rand()
val target = Tensor[Float](3).rand()
val output = criterion.forward(input, target)
```
output is
```
output: Float = 0.33056465
```

**Python example:**
```python
Python Code
```
