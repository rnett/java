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
import org.tensorflow.types.family.TType;

/**
 * Returns a Tensor stack of all keys in a tensor map.
 * <p>
 * input_handle: the input map
 * keys: the returned Tensor of all keys in the map
 * 
 * @param <T> data type for {@code keys()} output
 */
@Operator
public final class TensorMapStackKeys<T extends TType> extends RawOp implements Operand<T> {
  
  /**
   * Factory method to create a class wrapping a new TensorMapStackKeys operation.
   * 
   * @param scope current scope
   * @param inputHandle 
   * @param keyDtype 
   * @return a new instance of TensorMapStackKeys
   */
  @Endpoint(describeByClass = true)
  public static <T extends TType> TensorMapStackKeys<T> create(Scope scope, Operand<?> inputHandle, DataType<T> keyDtype) {
    OperationBuilder opBuilder = scope.env().opBuilder("TensorMapStackKeys", scope.makeOpName("TensorMapStackKeys"));
    opBuilder.addInput(inputHandle.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    opBuilder.setAttr("key_dtype", keyDtype);
    return new TensorMapStackKeys<T>(opBuilder.build());
  }
  
  /**
   */
  public Output<T> keys() {
    return keys;
  }
  
  @Override
  public Output<T> asOutput() {
    return keys;
  }
  
  /** The name of this op, as known by TensorFlow core engine */
  public static final String OP_NAME = "TensorMapStackKeys";
  
  private Output<T> keys;
  
  private TensorMapStackKeys(Operation operation) {
    super(operation);
    int outputIdx = 0;
    keys = operation.output(outputIdx++);
  }
}
