/*
  Copyright 2021 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================
 */
package org.tensorflow.op;

import static org.tensorflow.internal.c_api.global.tensorflow.StatusFromTF_Status;

import java.util.List;
import org.bytedeco.javacpp.PointerScope;
import org.tensorflow.GradientAdapterHelpers;
import org.tensorflow.Graph;
import org.tensorflow.GraphOperation;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.internal.c_api.GradFunc;
import org.tensorflow.internal.c_api.NativeOperation;
import org.tensorflow.internal.c_api.NativeOutput;
import org.tensorflow.internal.c_api.NativeStatus;
import org.tensorflow.internal.c_api.TF_Scope;
import org.tensorflow.internal.c_api.TF_Status;

/**
 * A native adapter for {@link RawCustomGradient}.
 */
public class RawGradientAdapter extends GradFunc {

  private final RawCustomGradient gradient;

  public RawGradientAdapter(RawCustomGradient gradient) {
    this.gradient = gradient;
  }

  @Override
  public NativeStatus call(TF_Scope scope, NativeOperation op, NativeOutput grad_inputs,
      NativeOutput grad_outputs) {
    try (PointerScope pointerScope = new PointerScope()) {
      Graph g = Graph.findGraphForPointer(scope.graph());
      if (g == null) {
        throw new IllegalStateException("No graph found for native gradient scope.");
      }

      Scope nativeScope = new NativeScope(scope, g);
      Ops tf = new Ops(nativeScope);

      List<Output<?>> gradInputs = GradientAdapterHelpers.fromNativeOutputs(g, grad_inputs);

      GraphOperation operation = GradientAdapterHelpers.getGraphOp(g, op.node());

      List<Operand<?>> gradOutputs = gradient.call(tf, operation, gradInputs);
      GradientAdapterHelpers.putToNativeOutputs(gradOutputs, grad_outputs);
    }
    return StatusFromTF_Status(TF_Status.newStatus());
  }
}
