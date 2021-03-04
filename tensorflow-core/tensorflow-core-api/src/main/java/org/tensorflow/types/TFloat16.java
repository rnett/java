/*
 *  Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */

package org.tensorflow.types;

import java.util.function.Consumer;
import org.tensorflow.Tensor;
import org.tensorflow.TensorScope;
import org.tensorflow.exceptions.TensorFlowException;
import org.tensorflow.internal.types.TFloat16Mapper;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.types.annotation.TensorType;
import org.tensorflow.types.family.TFloating;

/**
 * IEEE-754 half-precision 16-bit float tensor type.
 *
 * <p>Since there is no floating-point type that fits in 16 bits in Java, a conversion (with
 * potentially a precision loss) is required for each 32 bits value written or read on a tensor of this type from the
 * JVM. Therefore, if a lot of I/O operations are to be expected on a tensor, performances will be improved by working
 * with {@link TFloat32} or {@link TFloat64} data types whenever possible.
 *
 * <p>Also, {@code TFloat16} tensors normally perform better if they are located in GPU memory since
 * most CPUs do not support this format natively. For CPU computation on 16-bit floats, the {@link TBfloat16} tensor
 * type might be a better option.
 */
@TensorType(dataType = DataType.DT_HALF, byteSize = 2, mapperClass = TFloat16Mapper.class)
public interface TFloat16 extends FloatNdArray, TFloating {

  /**
   * Allocates a new tensor for storing a single float value.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param value float to store in the new tensor
   * @return the new tensor
   */
  static TFloat16 scalarOf(TensorScope scope, float value) {
    return Tensor.of(scope, TFloat16.class, Shape.scalar(), data -> data.setFloat(value));
  }

  /**
   * Allocates a new tensor for storing a vector of floats.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param values floats to store in the new tensor
   * @return the new tensor
   */
  static TFloat16 vectorOf(TensorScope scope, float... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return Tensor.of(scope, TFloat16.class, Shape.of(values.length), data -> StdArrays.copyTo(values, data));
  }

  /**
   * Allocates a new tensor which is a copy of a given array of floats.
   *
   * <p>The tensor will have the same shape as the source array and its data will be copied.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param src the source array giving the shape and data to the new tensor
   * @return the new tensor
   */
  static TFloat16 tensorOf(TensorScope scope, NdArray<Float> src) {
    return Tensor.of(scope, TFloat16.class, src.shape(), src::copyTo);
  }

  /**
   * Allocates a new tensor of the given shape.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param shape shape of the tensor to allocate
   * @return the new tensor
   */
  static TFloat16 tensorOf(TensorScope scope, Shape shape) {
    return Tensor.of(scope, TFloat16.class, shape);
  }

  /**
   * Allocates a new tensor of the given shape, initialized with the provided data.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param shape shape of the tensor to allocate
   * @param data buffer of floats to initialize the tensor with
   * @return the new tensor
   */
  static TFloat16 tensorOf(TensorScope scope, Shape shape, FloatDataBuffer data) {
    return Tensor.of(scope, TFloat16.class, shape, d -> d.write(data));
  }

  /**
   * Allocates a new tensor of the given shape and initialize its data.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param shape shape of the tensor to allocate
   * @param dataInit tensor data initializer
   * @return the new tensor
   * @throws TensorFlowException if the tensor cannot be allocated or initialized
   */
  static TFloat16 tensorOf(TensorScope scope, Shape shape, Consumer<TFloat16> dataInit) {
    return Tensor.of(scope, TFloat16.class, shape, dataInit);
  }
}
