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
 * Returns a map that is the 'input_handle' with the given key-value pair inserted.
 * <p>
 * input_handle: the original map
 * output_handle: the map with key and value inserted
 * key: the key to be inserted
 * value: the value to be inserted
 */
@Operator
public final class TensorMapInsert extends RawOp implements Operand<TType> {
  
  /**
   * Factory method to create a class wrapping a new TensorMapInsert operation.
   * 
   * @param scope current scope
   * @param inputHandle 
   * @param key 
   * @param value 
   * @return a new instance of TensorMapInsert
   */
  @Endpoint(describeByClass = true)
  public static <T extends TType, U extends TType> TensorMapInsert create(Scope scope, Operand<?> inputHandle, Operand<T> key, Operand<U> value) {
    OperationBuilder opBuilder = scope.env().opBuilder("TensorMapInsert", scope.makeOpName("TensorMapInsert"));
    opBuilder.addInput(inputHandle.asOutput());
    opBuilder.addInput(key.asOutput());
    opBuilder.addInput(value.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    return new TensorMapInsert(opBuilder.build());
  }
  
  /**
   */
  public Output<?> outputHandle() {
    return outputHandle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<TType> asOutput() {
    return (Output<TType>) outputHandle;
  }
  
  /** The name of this op, as known by TensorFlow core engine */
  public static final String OP_NAME = "TensorMapInsert";
  
  private Output<?> outputHandle;
  
  private TensorMapInsert(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputHandle = operation.output(outputIdx++);
  }
}
