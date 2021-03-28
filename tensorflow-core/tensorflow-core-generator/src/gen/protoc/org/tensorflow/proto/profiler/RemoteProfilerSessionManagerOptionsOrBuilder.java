// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/profiler/profiler_options.proto

package org.tensorflow.proto.profiler;

public interface RemoteProfilerSessionManagerOptionsOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.RemoteProfilerSessionManagerOptions)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Options for each local profiler.
   * </pre>
   *
   * <code>.tensorflow.ProfileOptions profiler_options = 1;</code>
   */
  boolean hasProfilerOptions();
  /**
   * <pre>
   * Options for each local profiler.
   * </pre>
   *
   * <code>.tensorflow.ProfileOptions profiler_options = 1;</code>
   */
  org.tensorflow.proto.profiler.ProfileOptions getProfilerOptions();
  /**
   * <pre>
   * Options for each local profiler.
   * </pre>
   *
   * <code>.tensorflow.ProfileOptions profiler_options = 1;</code>
   */
  org.tensorflow.proto.profiler.ProfileOptionsOrBuilder getProfilerOptionsOrBuilder();

  /**
   * <pre>
   * List of servers to profile. Supported formats: host:port.
   * </pre>
   *
   * <code>repeated string service_addresses = 2;</code>
   */
  java.util.List<java.lang.String>
      getServiceAddressesList();
  /**
   * <pre>
   * List of servers to profile. Supported formats: host:port.
   * </pre>
   *
   * <code>repeated string service_addresses = 2;</code>
   */
  int getServiceAddressesCount();
  /**
   * <pre>
   * List of servers to profile. Supported formats: host:port.
   * </pre>
   *
   * <code>repeated string service_addresses = 2;</code>
   */
  java.lang.String getServiceAddresses(int index);
  /**
   * <pre>
   * List of servers to profile. Supported formats: host:port.
   * </pre>
   *
   * <code>repeated string service_addresses = 2;</code>
   */
  com.google.protobuf.ByteString
      getServiceAddressesBytes(int index);

  /**
   * <pre>
   * Unix timestamp of when the session was started.
   * </pre>
   *
   * <code>uint64 session_creation_timestamp_ns = 3;</code>
   */
  long getSessionCreationTimestampNs();

  /**
   * <pre>
   * Maximum time (in milliseconds) a profiling session manager waits for all
   * profilers to finish after issuing gRPC request. If value is 0, session
   * continues until interrupted. Otherwise, value must be greater than
   * profiler_options.duration_ms.
   * </pre>
   *
   * <code>uint64 max_session_duration_ms = 4;</code>
   */
  long getMaxSessionDurationMs();

  /**
   * <pre>
   * Start of profiling is delayed by this much (in milliseconds).
   * </pre>
   *
   * <code>uint64 delay_ms = 5;</code>
   */
  long getDelayMs();
}
