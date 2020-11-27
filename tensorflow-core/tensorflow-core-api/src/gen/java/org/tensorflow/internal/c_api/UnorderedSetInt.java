// Targeted by JavaCPP version 1.5.4: DO NOT EDIT THIS FILE

package org.tensorflow.internal.c_api;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.tensorflow.internal.c_api.global.tensorflow.*;

@Name("std::unordered_set<int>") @Properties(inherit = org.tensorflow.internal.c_api.presets.tensorflow.class)
public class UnorderedSetInt extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UnorderedSetInt(Pointer p) { super(p); }
    public UnorderedSetInt()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef UnorderedSetInt put(@ByRef UnorderedSetInt x);

    public boolean empty() { return size() == 0; }
    public native long size();

    public native void insert(int value);
    public native void erase(int value);
    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator ++") @ByRef Iterator increment();
        public native @Name("operator ==") boolean equals(@ByRef Iterator it);
        public native @Name("operator *") int get();
    }
}

