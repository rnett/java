// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/debug_event.proto

package org.tensorflow.proto.util;

public interface DebuggedDeviceOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.DebuggedDevice)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Name of the device.
   * </pre>
   *
   * <code>string device_name = 1;</code>
   */
  java.lang.String getDeviceName();
  /**
   * <pre>
   * Name of the device.
   * </pre>
   *
   * <code>string device_name = 1;</code>
   */
  com.google.protobuf.ByteString
      getDeviceNameBytes();

  /**
   * <pre>
   * A debugger-generated ID for the device. Guaranteed to be unique within
   * the scope of the debugged TensorFlow program, including single-host and
   * multi-host settings.
   * TODO(cais): Test the uniqueness guarantee in multi-host settings.
   * </pre>
   *
   * <code>int32 device_id = 2;</code>
   */
  int getDeviceId();
}
