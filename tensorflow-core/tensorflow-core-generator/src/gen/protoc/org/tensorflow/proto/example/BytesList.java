// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/example/feature.proto

package org.tensorflow.proto.example;

/**
 * <pre>
 * Containers to hold repeated fundamental values.
 * </pre>
 *
 * Protobuf type {@code tensorflow.BytesList}
 */
public  final class BytesList extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.BytesList)
    BytesListOrBuilder {
private static final long serialVersionUID = 0L;
  // Use BytesList.newBuilder() to construct.
  private BytesList(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private BytesList() {
    value_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new BytesList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private BytesList(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 10: {
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              value_ = new java.util.ArrayList<com.google.protobuf.ByteString>();
              mutable_bitField0_ |= 0x00000001;
            }
            value_.add(input.readBytes());
            break;
          }
          default: {
            if (!parseUnknownField(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      if (((mutable_bitField0_ & 0x00000001) != 0)) {
        value_ = java.util.Collections.unmodifiableList(value_); // C
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.proto.example.FeatureProtos.internal_static_tensorflow_BytesList_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.proto.example.FeatureProtos.internal_static_tensorflow_BytesList_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.proto.example.BytesList.class, org.tensorflow.proto.example.BytesList.Builder.class);
  }

  public static final int VALUE_FIELD_NUMBER = 1;
  private java.util.List<com.google.protobuf.ByteString> value_;
  /**
   * <code>repeated bytes value = 1;</code>
   */
  public java.util.List<com.google.protobuf.ByteString>
      getValueList() {
    return value_;
  }
  /**
   * <code>repeated bytes value = 1;</code>
   */
  public int getValueCount() {
    return value_.size();
  }
  /**
   * <code>repeated bytes value = 1;</code>
   */
  public com.google.protobuf.ByteString getValue(int index) {
    return value_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < value_.size(); i++) {
      output.writeBytes(1, value_.get(i));
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < value_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeBytesSizeNoTag(value_.get(i));
      }
      size += dataSize;
      size += 1 * getValueList().size();
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tensorflow.proto.example.BytesList)) {
      return super.equals(obj);
    }
    org.tensorflow.proto.example.BytesList other = (org.tensorflow.proto.example.BytesList) obj;

    if (!getValueList()
        .equals(other.getValueList())) return false;
    if (!unknownFields.equals(other.unknownFields)) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (getValueCount() > 0) {
      hash = (37 * hash) + VALUE_FIELD_NUMBER;
      hash = (53 * hash) + getValueList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.proto.example.BytesList parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.proto.example.BytesList parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.proto.example.BytesList parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.proto.example.BytesList parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.tensorflow.proto.example.BytesList prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * <pre>
   * Containers to hold repeated fundamental values.
   * </pre>
   *
   * Protobuf type {@code tensorflow.BytesList}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.BytesList)
      org.tensorflow.proto.example.BytesListOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.proto.example.FeatureProtos.internal_static_tensorflow_BytesList_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.proto.example.FeatureProtos.internal_static_tensorflow_BytesList_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.proto.example.BytesList.class, org.tensorflow.proto.example.BytesList.Builder.class);
    }

    // Construct using org.tensorflow.proto.example.BytesList.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      value_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.proto.example.FeatureProtos.internal_static_tensorflow_BytesList_descriptor;
    }

    @java.lang.Override
    public org.tensorflow.proto.example.BytesList getDefaultInstanceForType() {
      return org.tensorflow.proto.example.BytesList.getDefaultInstance();
    }

    @java.lang.Override
    public org.tensorflow.proto.example.BytesList build() {
      org.tensorflow.proto.example.BytesList result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tensorflow.proto.example.BytesList buildPartial() {
      org.tensorflow.proto.example.BytesList result = new org.tensorflow.proto.example.BytesList(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) != 0)) {
        value_ = java.util.Collections.unmodifiableList(value_);
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.value_ = value_;
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tensorflow.proto.example.BytesList) {
        return mergeFrom((org.tensorflow.proto.example.BytesList)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.proto.example.BytesList other) {
      if (other == org.tensorflow.proto.example.BytesList.getDefaultInstance()) return this;
      if (!other.value_.isEmpty()) {
        if (value_.isEmpty()) {
          value_ = other.value_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureValueIsMutable();
          value_.addAll(other.value_);
        }
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.tensorflow.proto.example.BytesList parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.proto.example.BytesList) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<com.google.protobuf.ByteString> value_ = java.util.Collections.emptyList();
    private void ensureValueIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        value_ = new java.util.ArrayList<com.google.protobuf.ByteString>(value_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public java.util.List<com.google.protobuf.ByteString>
        getValueList() {
      return ((bitField0_ & 0x00000001) != 0) ?
               java.util.Collections.unmodifiableList(value_) : value_;
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public int getValueCount() {
      return value_.size();
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public com.google.protobuf.ByteString getValue(int index) {
      return value_.get(index);
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public Builder setValue(
        int index, com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureValueIsMutable();
      value_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public Builder addValue(com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureValueIsMutable();
      value_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public Builder addAllValue(
        java.lang.Iterable<? extends com.google.protobuf.ByteString> values) {
      ensureValueIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, value_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated bytes value = 1;</code>
     */
    public Builder clearValue() {
      value_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:tensorflow.BytesList)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.BytesList)
  private static final org.tensorflow.proto.example.BytesList DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.proto.example.BytesList();
  }

  public static org.tensorflow.proto.example.BytesList getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<BytesList>
      PARSER = new com.google.protobuf.AbstractParser<BytesList>() {
    @java.lang.Override
    public BytesList parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new BytesList(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<BytesList> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<BytesList> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tensorflow.proto.example.BytesList getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

