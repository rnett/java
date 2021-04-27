// Targeted by JavaCPP version 1.5.5: DO NOT EDIT THIS FILE

package org.tensorflow.internal.c_api;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.tensorflow.internal.c_api.global.tensorflow.*;

@Name("std::vector<tensorflow::Output>") @Properties(inherit = org.tensorflow.internal.c_api.presets.tensorflow.class)
public class NativeOutputVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NativeOutputVector(Pointer p) { super(p); }
    public NativeOutputVector(NativeOutput value) { this(1); put(0, value); }
    public NativeOutputVector(NativeOutput ... array) { this(array.length); put(array); }
    public NativeOutputVector()       { allocate();  }
    public NativeOutputVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator =") @ByRef NativeOutputVector put(@ByRef NativeOutputVector x);

    public boolean empty() { return size() == 0; }
    public native long size();
    public void clear() { resize(0); }
    public native void resize(@Cast("size_t") long n);

    @Index(function = "at") public native @ByRef NativeOutput get(@Cast("size_t") long i);
    public native NativeOutputVector put(@Cast("size_t") long i, NativeOutput value);

    public native @ByVal Iterator insert(@ByVal Iterator pos, @ByRef NativeOutput value);
    public native @ByVal Iterator erase(@ByVal Iterator pos);
    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator ++") @ByRef Iterator increment();
        public native @Name("operator ==") boolean equals(@ByRef Iterator it);
        public native @Name("operator *") @ByRef @Const NativeOutput get();
    }

    public NativeOutput[] get() {
        NativeOutput[] array = new NativeOutput[size() < Integer.MAX_VALUE ? (int)size() : Integer.MAX_VALUE];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
    @Override public String toString() {
        return java.util.Arrays.toString(get());
    }

    public NativeOutput pop_back() {
        long size = size();
        NativeOutput value = get(size - 1);
        resize(size - 1);
        return value;
    }
    public NativeOutputVector push_back(NativeOutput value) {
        long size = size();
        resize(size + 1);
        return put(size, value);
    }
    public NativeOutputVector put(NativeOutput value) {
        if (size() != 1) { resize(1); }
        return put(0, value);
    }
    public NativeOutputVector put(NativeOutput ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

