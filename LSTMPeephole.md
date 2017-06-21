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
val batchSize = 1
               
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
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.35383725	-0.22476536	-0.46047324	-0.26038578	-0.21095484	
0.3409024	-0.22834192	-0.41133574	-0.27646995	-0.23721263	
0.39881697	-0.18804908	-0.48271912	-0.29778507	-0.14873621	
0.43038777	-0.16956224	-0.46273726	-0.30802295	-0.12813234	
0.32592735	-0.24277578	-0.42178982	-0.27876818	-0.23236775	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

```
**Python example:**
```python
Python Code
```
