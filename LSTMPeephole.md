## LSTMPeephole ##

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
import com.intel.analytics.bigdl.utils.RandomGenerator._

val hiddenSize = 4
val inputSize = 6
val outputSize = 5
val seqLength = 5
               
val input = Tensor[Float](Array(batchSize, seqLength, inputSize))
for (b <- 1 to batchSize) {
  for (i <- 1 to seqLength) {
    val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
    input.setValue(b, i, rdmInput, 1.0f)
  }
}

val rec = Recurrent[Float](hiddenSize)
val model = Sequential[Float]().add(rec.add(LSTMPeephole[Float](inputSize, hiddenSize))).add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))
val output = model.forward(input).toTensor
```
output is
```


```
**Python example:**
```python
Python Code
```
