## LogSoftMax ##

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

val module = LogSoftMax[Float]()
val input = Tensor[Float](4, 10).rand()
val output = module.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
-2.5519505	-2.842981	-2.7291634	-2.5205712	-2.21346	-2.0949225	-2.0828898	-2.0750494	-1.98947	-2.3142338	
-2.4164822	-2.2851913	-2.344031	-1.9481893	-2.3196316	-1.9729276	-2.7517028	-2.2612143	-2.7797568	-2.2722216	
-2.5480394	-2.5438929	-2.1472383	-1.8264041	-2.4599571	-2.3786807	-2.347884	-1.8615696	-2.6476033	-2.726078	
-2.2741008	-2.0731382	-2.500853	-2.3554156	-2.2530231	-2.014162	-2.5651312	-2.1602802	-2.4301133	-2.5808542	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x10]
```

**Python example:**
```python
Python Code
```
