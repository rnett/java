package org.tensorflow.internal.c_api;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.tensorflow.internal.c_api.presets.tensorflow.class)
public abstract class AbstractTapeVSpace extends Pointer {

    public AbstractTapeVSpace(Pointer p) { super(p); }

}
