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
package org.tensorflow.op.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Scope;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

public class BooleanMaskTest {
  @Test
  public void testBooleanMask(){
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);

      Operand<TInt32> input = Constant.arrayOf(scope, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
      Operand<TInt32> input2 = ExpandDims.create(scope, input, Constant.scalarOf(scope, 0));

      Operand<TBool> mask = Constant.arrayOf(scope, true, true, false, false, true, true, true, false, false, false);

      Operand<TInt32> output1 = BooleanMask.create(scope, input, mask);
      Operand<TInt32> output2 = BooleanMask.create(scope, input2, mask, BooleanMask.axis(1));

      try (TFloat32 result = (TFloat32) sess.runner().fetch(output1).run().get(0)) {
        // expected shape from Python tensorflow
        assertEquals(Shape.of(5), result.shape());
        assertEquals(result.getFloat(0), 0);
        assertEquals(result.getFloat(1), 1);
        assertEquals(result.getFloat(2), 4);
        assertEquals(result.getFloat(3), 5);
        assertEquals(result.getFloat(4), 6);
      }

      try (TFloat32 result = (TFloat32) sess.runner().fetch(output2).run().get(0)) {
        // expected shape from Python tensorflow
        assertEquals(Shape.of(5), result.shape());
        assertEquals(result.getFloat(0), 0);
        assertEquals(result.getFloat(1), 1);
        assertEquals(result.getFloat(2), 4);
        assertEquals(result.getFloat(3), 5);
        assertEquals(result.getFloat(4), 6);
      }
    }
  }
}
