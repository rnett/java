package org.tensorflow.internal.c_api;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.tensorflow.internal.c_api.presets.tensorflow.class)
public class AbstractTFE_TensorHandleList extends Pointer {

    public AbstractTFE_TensorHandleList(Pointer p) { super(p); }

}
