/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

// This class has been generated, DO NOT EDIT!

package org.tensorflow.op.core;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.RawOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Endpoint;
import org.tensorflow.op.annotation.Operator;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 * Outputs deterministic pseudorandom values from a truncated normal distribution.
 * <p>
 * The generated values follow a normal distribution with mean 0 and standard
 * deviation 1, except that values whose magnitude is more than 2 standard
 * deviations from the mean are dropped and re-picked.
 * <p>
 * The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.
 * 
 * @param <U> data type for {@code output()} output
 */
public final class StatelessTruncatedNormalV2<U extends TNumber> extends RawOp implements Operand<U> {
  
  /**
   * Factory method to create a class wrapping a new StatelessTruncatedNormalV2 operation.
   * 
   * @param scope current scope
   * @param shape The shape of the output tensor.
   * @param key Key for the counter-based RNG algorithm (shape uint64[1]).
   * @param counter Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
   * @param alg The RNG algorithm (shape int32[]).
   * @param dtype The type of the output.
   * @return a new instance of StatelessTruncatedNormalV2
   */
  @Endpoint(describeByClass = true)
  public static <U extends TNumber, T extends TNumber> StatelessTruncatedNormalV2<U> create(Scope scope, Operand<T> shape, Operand<?> key, Operand<?> counter, Operand<TInt32> alg, DataType<U> dtype) {
    OperationBuilder opBuilder = scope.env().opBuilder("StatelessTruncatedNormalV2", scope.makeOpName("StatelessTruncatedNormalV2"));
    opBuilder.addInput(shape.asOutput());
    opBuilder.addInput(key.asOutput());
    opBuilder.addInput(counter.asOutput());
    opBuilder.addInput(alg.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    opBuilder.setAttr("dtype", dtype);
    return new StatelessTruncatedNormalV2<U>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class wrapping a new StatelessTruncatedNormalV2 operation using default output types.
   * 
   * @param scope current scope
   * @param shape The shape of the output tensor.
   * @param key Key for the counter-based RNG algorithm (shape uint64[1]).
   * @param counter Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
   * @param alg The RNG algorithm (shape int32[]).
   * @return a new instance of StatelessTruncatedNormalV2
   */
  @Endpoint(describeByClass = true)
  public static <T extends TNumber> StatelessTruncatedNormalV2<TFloat32> create(Scope scope, Operand<T> shape, Operand<?> key, Operand<?> counter, Operand<TInt32> alg) {
    return create(scope, shape, key, counter, alg, TFloat32.DTYPE);
  }
  
  /**
   * Random values with specified shape.
   */
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  /** The name of this op, as known by TensorFlow core engine */
  public static final String OP_NAME = "StatelessTruncatedNormalV2";
  
  private Output<U> output;
  
  private StatelessTruncatedNormalV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
