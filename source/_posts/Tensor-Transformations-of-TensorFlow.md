---
title: Tensor Transformations of TensorFlow
date: 2018-04-09 00:03:51
tags: Tensor Transformations
---
# 数据类型转换
## 1. tf.string_to_number
```
tf.string_to_number(
    string_tensor,
        out_type=tf.float32,
	    name=None
	    )
	    ```
	    将输入张量tensor中的每个字符串转换成指定的数字类型。

	    参数：

	    * string_tensor ：一个string类型的tensor。
	    * out_type：转换数据的数据类型；可选参数，默认是tf.float32，还可以设置 tf.float32, tf.float64, tf.int32, tf.int64
	    * name: 操作名，可选参数，默认是None

	    返回值：

	    返回一个out_type类型的tensor

	    ## 2. tf.to_double
	    ```
	    tf.to_double(
	        x,
		    name='ToDouble'
		    )
		    ```

		    转换一个tensor数据为float64。

		    参数：

		    * x: 一个tensor类型或SparseTensor类型
		    * name: 可选参数，操作名。

		    返回值：

		    返回float64, 如果不能转换返回TypeError 。

		    ## 3. tf.to_float
		    ```
		    tf.to_float(
		        x,
			    name='ToFloat'
			    )
			    ```

			    与tf.to_double差不多类似，返回一个float32，如果错误返回TypeError.

			    ## 4. tf.to_bfloat16
			    ```
			    tf.to_bfloat16(
			        x,
				    name='ToBFloat16'
				    )
				    ```
				    参与与前面2个相似，返回bfloat16。

				    ## 4. tf.to_int32
				    ```
				    tf.to_int32(
				        x,
					    name='ToInt32'
					    )
					    ```

					    参数与前面相同，将tensor x 转换为int32类型。

					    ## 5. tf.to_int64
					    ```
					    tf.to_int64(
					        x,
						    name='ToInt64'
						    )
						    ```

						    参数与前面相同，返回int64类型tensor。

						    ## 6. tf.cast
						    ```
						    tf.cast(
						        x,
							    dtype,
							        name=None
								)
								```
								将tensor x转换为dtype类型的tensor， shape相同。

								example:
								```
								x = tf.constant([1.8, 2.2], dtype=tf.float32)
								tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
								```

								## 7. tf.bitcast
								```
								tf.bitcast(
								    input,
								        type,
									    name=None
									    )
									    ```
									    <!-- [TODO]: 
									    If the input datatype T is larger than the output datatype type then the shape changes from [...] to [..., sizeof(T)/sizeof(type)].

									    [TODO]: 
									    If T is smaller than type, the operator requires that the rightmost dimension be equal to sizeof(type)/sizeof(T). The shape then goes from [..., sizeof(type)/sizeof(T)] to [...].

									    [TODO]: 
									    NOTE: Bitcast is implemented as a low-level cast, so machines with different endian orderings will give different results -->

									    参数：

									    * input：支持类型：bfloat16, float32, float64, int64, int32, uint8, uint16, int8, int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, half

									    * type: 一个tf.DType类型，支持：tf.bfloat16, tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int8, tf.int16, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32, tf.half

									    * name: 操作名，可选参数。

									    返回值：

									    返回参数type设置的类型tensor数据。

									    ## 7. tf.saturate_cast
									    ```
									    tf.saturate_cast(
									        value,
										    dtype,
										        name=None
											)
											```

											比较安全的类型转换，相比tf.bitcast不会有任何的缩放，使用之前应该保证数据长度shape。把value数据转换为类型dtype类型的数据。操作名name可选参数




