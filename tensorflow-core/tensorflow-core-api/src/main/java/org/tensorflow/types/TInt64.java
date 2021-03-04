/*
 *  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import org.tensorflow.internal.types.TInt64Mapper;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.types.annotation.TensorType;
import org.tensorflow.types.family.TIntegral;

/**
 * 64-bit signed integer tensor type.
 */
@TensorType(dataType = DataType.DT_INT64, byteSize = 8, mapperClass = TInt64Mapper.class)
public interface TInt64 extends LongNdArray, TIntegral {

  /**
   * Allocates a new tensor for storing a single long value.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param value long to store in the new tensor
   * @return the new tensor
   */
  static TInt64 scalarOf(TensorScope scope, long value) {
    return Tensor.of(scope, TInt64.class, Shape.scalar(), data -> data.setLong(value));
  }

  /**
   * Allocates a new tensor for storing a vector of longs.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param values longs to store in the new tensor
   * @return the new tensor
   */
  static TInt64 vectorOf(TensorScope scope, long... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return Tensor.of(scope, TInt64.class, Shape.of(values.length), data -> StdArrays.copyTo(values, data));
  }

  /**
   * Allocates a new tensor which is a copy of a given array of longs.
   *
   * <p>The tensor will have the same shape as the source array and its data will be copied.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param src the source array giving the shape and data to the new tensor
   * @return the new tensor
   */
  static TInt64 tensorOf(TensorScope scope, NdArray<Long> src) {
    return Tensor.of(scope, TInt64.class, src.shape(), src::copyTo);
  }

  /**
   * Allocates a new tensor of the given shape.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param shape shape of the tensor to allocate
   * @return the new tensor
   */
  static TInt64 tensorOf(TensorScope scope, Shape shape) {
    return Tensor.of(scope, TInt64.class, shape);
  }

  /**
   * Allocates a new tensor of the given shape, initialized with the provided data.
   *
   * @param scope the {@link TensorScope} to create the tensor in
   * @param shape shape of the tensor to allocate
   * @param data buffer of longs to initialize the tensor with
   * @return the new tensor
   */
  static TInt64 tensorOf(TensorScope scope, Shape shape, LongDataBuffer data) {
    return Tensor.of(scope, TInt64.class, shape, d -> d.write(data));
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
  static TInt64 tensorOf(TensorScope scope, Shape shape, Consumer<TInt64> dataInit) {
    return Tensor.of(scope, TInt64.class, shape, dataInit);
  }
}
