// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/attr_value.proto

package org.tensorflow.proto.framework;

public final class AttrValueProtos {
  private AttrValueProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_AttrValue_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_AttrValue_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_AttrValue_ListValue_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_AttrValue_ListValue_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_NameAttrList_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_NameAttrList_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_NameAttrList_AttrEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_NameAttrList_AttrEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    String[] descriptorData = {
        "\n*tensorflow/core/framework/attr_value.p" +
            "roto\022\ntensorflow\032&tensorflow/core/framew" +
            "ork/tensor.proto\032,tensorflow/core/framew" +
            "ork/tensor_shape.proto\032%tensorflow/core/" +
            "framework/types.proto\"\246\004\n\tAttrValue\022\013\n\001s" +
            "\030\002 \001(\014H\000\022\013\n\001i\030\003 \001(\003H\000\022\013\n\001f\030\004 \001(\002H\000\022\013\n\001b\030"
            +
            "\005 \001(\010H\000\022$\n\004type\030\006 \001(\0162\024.tensorflow.DataT" +
            "ypeH\000\022-\n\005shape\030\007 \001(\0132\034.tensorflow.Tensor" +
            "ShapeProtoH\000\022)\n\006tensor\030\010 \001(\0132\027.tensorflo" +
            "w.TensorProtoH\000\022/\n\004list\030\001 \001(\0132\037.tensorfl" +
      "ow.AttrValue.ListValueH\000\022(\n\004func\030\n \001(\0132\030" +
      ".tensorflow.NameAttrListH\000\022\025\n\013placeholde" +
      "r\030\t \001(\tH\000\032\351\001\n\tListValue\022\t\n\001s\030\002 \003(\014\022\r\n\001i\030" +
      "\003 \003(\003B\002\020\001\022\r\n\001f\030\004 \003(\002B\002\020\001\022\r\n\001b\030\005 \003(\010B\002\020\001\022" +
      "&\n\004type\030\006 \003(\0162\024.tensorflow.DataTypeB\002\020\001\022" +
      "+\n\005shape\030\007 \003(\0132\034.tensorflow.TensorShapeP" +
      "roto\022\'\n\006tensor\030\010 \003(\0132\027.tensorflow.Tensor" +
      "Proto\022&\n\004func\030\t \003(\0132\030.tensorflow.NameAtt" +
      "rListB\007\n\005value\"\222\001\n\014NameAttrList\022\014\n\004name\030" +
      "\001 \001(\t\0220\n\004attr\030\002 \003(\0132\".tensorflow.NameAtt" +
      "rList.AttrEntry\032B\n\tAttrEntry\022\013\n\003key\030\001 \001(" +
      "\t\022$\n\005value\030\002 \001(\0132\025.tensorflow.AttrValue:" +
      "\0028\001B\211\001\n\036org.tensorflow.proto.frameworkB\017" +
      "AttrValueProtosP\001ZQgithub.com/tensorflow" +
      "/tensorflow/tensorflow/go/core/framework" +
      "/attr_value_go_proto\370\001\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[]{
            TensorProtos.getDescriptor(),
            TensorShapeProtos.getDescriptor(),
            TypesProtos.getDescriptor(),
        });
    internal_static_tensorflow_AttrValue_descriptor =
        getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_AttrValue_fieldAccessorTable = new
        com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_AttrValue_descriptor,
        new String[]{"S", "I", "F", "B", "Type", "Shape", "Tensor", "List", "Func", "Placeholder", "Value",});
    internal_static_tensorflow_AttrValue_ListValue_descriptor =
        internal_static_tensorflow_AttrValue_descriptor.getNestedTypes().get(0);
    internal_static_tensorflow_AttrValue_ListValue_fieldAccessorTable = new
        com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_AttrValue_ListValue_descriptor,
        new String[]{"S", "I", "F", "B", "Type", "Shape", "Tensor", "Func",});
    internal_static_tensorflow_NameAttrList_descriptor =
        getDescriptor().getMessageTypes().get(1);
    internal_static_tensorflow_NameAttrList_fieldAccessorTable = new
        com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_NameAttrList_descriptor,
        new String[]{"Name", "Attr",});
    internal_static_tensorflow_NameAttrList_AttrEntry_descriptor =
        internal_static_tensorflow_NameAttrList_descriptor.getNestedTypes().get(0);
    internal_static_tensorflow_NameAttrList_AttrEntry_fieldAccessorTable = new
        com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_NameAttrList_AttrEntry_descriptor,
        new String[]{"Key", "Value",});
    TensorProtos.getDescriptor();
    TensorShapeProtos.getDescriptor();
    TypesProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
