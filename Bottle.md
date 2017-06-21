## Bottle ##

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

val module = Bottle[Float](Linear[Float](10, 2), 2, 2)
module.add(Linear(10, 2))
val input = Tensor[Float](4, 5, 10).rand()
val output = module.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.6945294	-0.27953064	
0.6268389	-0.7294409	
0.69834805	-0.42664433	
0.70373046	-0.4026499	
0.66308194	-0.6336497	

(2,.,.) =
0.76823425	-0.57179654	
0.54741347	-0.5171715	
0.6170485	-0.48814133	
0.89729875	-0.5363091	
0.9383141	-0.63053	

(3,.,.) =
0.6869495	-0.6013391	
0.72504604	-0.44045419	
0.84359026	-0.51410943	
0.7153435	-0.783236	
0.8234116	-0.6176827	

(4,.,.) =
0.8869035	-0.51233184	
0.65199244	-0.48857856	
0.7880871	-0.7456757	
0.8663832	-0.22757408	
0.9411352	-0.8008182	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x5x2]
```

**Python example:**
```python
Python Code
```
