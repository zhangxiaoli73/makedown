## ValidationMethod ##

ValidationMethod is defined to evaluate the model.
This trait can be extended by user-defined method. Such as Top1Accuracy, Top5Accuracy.

Example code:

```
val output = Tensor(Storage(Array[Float](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = Tensor(Storage(Array[Float](4, 2, 1, 3, 2, 2, 2, 4)))

    val validation = new Top1Accuracy[Float]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
```
