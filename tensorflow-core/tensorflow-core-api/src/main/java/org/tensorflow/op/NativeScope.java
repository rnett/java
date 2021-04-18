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

import org.bytedeco.javacpp.PointerPointer;
import org.tensorflow.*;
import org.tensorflow.internal.c_api.TF_Operation;
import org.tensorflow.internal.c_api.TF_Scope;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public final class NativeScope implements IScope {

    @Override
    public ExecutionEnvironment env() {
        return graph;
    }

    @Override
    public NativeScope withSubScope(String childScopeName) {
        return new NativeScope(nativeScope.NewSubScope(childScopeName), graph);
    }

    @Override
    public NativeScope withName(String opName) {
        return new NativeScope(nativeScope, graph, opName);
    }

    @Override
    public NativeScope withNameAsSubScope(String defaultName) {
        return withSubScope(opName);
    }

    @Override
    public NativeScope withDevice(DeviceSpec deviceSpec) {
        return new NativeScope(nativeScope.WithDevice(deviceSpec.toString()), graph);
    }

    @Override
    public String makeOpName(String defaultName) {
        String name = opName != null ? opName : defaultName;
        return nativeScope.GetUniqueNameForOp(name);
    }

    @Override
    public NativeScope withControlDependencies(Iterable<Op> controls) {
        List<Op> controlDeps = StreamSupport.stream(controls.spliterator(), false).collect(Collectors.toList());
        PointerPointer<TF_Operation> ops = new PointerPointer<TF_Operation>(controlDeps.size());

        for(int i = 0 ; i < controlDeps.size() ; i++){
            Operation op = controlDeps.get(i).op();
            if(!(op instanceof GraphOperation))
                throw new IllegalArgumentException("Can only add graph ops as control dependencies");
            ops.put(i, (((GraphOperation) op).getUnsafeNativeHandle()));
        }

        return new NativeScope(nativeScope.WithControlDependencies(new TF_Operation(ops)), graph);
    }

    @Override
    public OperationBuilder apply(OperationBuilder builder) {
        return builder;
    }

    NativeScope(TF_Scope nativeScope, Graph graph){
        this(nativeScope, graph, null);
    }

    private NativeScope(TF_Scope nativeScope, Graph graph, String opName){
        this.graph = graph;
        this.nativeScope = nativeScope;
        this.opName = opName;
    }

    private final Graph graph;
    private final TF_Scope nativeScope;
    private final String opName;
}
