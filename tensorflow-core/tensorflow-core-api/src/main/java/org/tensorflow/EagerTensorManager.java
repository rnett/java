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
package org.tensorflow;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import org.bytedeco.javacpp.Pointer;
import org.tensorflow.internal.c_api.TFE_TensorHandle;

public class EagerTensorManager {

  private static MethodHandle trimMemory;

  static {
    try {
      // replace with MethodHandles.privateLookup w/ Java 9
      Method trimMem = Pointer.class.getDeclaredMethod("trimMemory");
      trimMem.setAccessible(true);
      trimMemory = MethodHandles.lookup().unreflect(trimMem);
      trimMem.setAccessible(false);
    } catch (NoSuchMethodException | IllegalAccessException e) {
      trimMemory = null;
    }
  }

  public static void register(TFE_TensorHandle tensor) {
    System.out
        .println("New tensor on device " + tensor.getDevice() + " with size " + Pointer.formatBytes(tensor.capacity()));
  }

  public static void cleanup() {
    System.gc();
    try {
      Thread.sleep(100);
    } catch (InterruptedException ignored) {
      Thread.currentThread().interrupt();
    }
//    Pointer.deallocateReferences();

    if (trimMemory != null) {
      try {
        trimMemory.invoke();
      } catch (Throwable ignored) {

      }
    }
  }
}
