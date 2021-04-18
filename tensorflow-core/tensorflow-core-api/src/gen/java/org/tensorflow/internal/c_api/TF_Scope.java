// Targeted by JavaCPP version 1.5.4: DO NOT EDIT THIS FILE

package org.tensorflow.internal.c_api;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.tensorflow.internal.c_api.global.tensorflow.*;


/** \addtogroup core
 *  \{
 <p>
 *  A {@code Scope} object represents a set of related TensorFlow ops that have the
 *  same properties such as a common name prefix.
 * 
 *  A Scope object is a container for TensorFlow Op properties. Op constructors
 *  get a Scope object as a mandatory first argument and the constructed op
 *  acquires the properties in the object.
 * 
 *  A simple example:
 * 
 *      using namespace ops;
 *      Scope root = Scope::NewRootScope();
 *      auto c1 = Const(root, { {1, 1} });
 *      auto m = MatMul(root, c1, { {41}, {1} });
 *      GraphDef gdef;
 *      Status s = root.ToGraphDef(&gdef);
 *      if (!s.ok()) { ... }
 * 
 *  Scope hierarchy:
 * 
 *  The Scope class provides various With<> functions that create a new scope.
 *  The new scope typically has one property changed while other properties are
 *  inherited from the parent scope.
 *  NewSubScope(name) method appends {@code name} to the prefix of names for ops
 *  created within the scope, and WithOpName() changes the suffix which
 *  otherwise defaults to the type of the op.
 * 
 *  Name examples:
 * 
 *      Scope root = Scope::NewRootScope();
 *      Scope linear = root.NewSubScope("linear");
 *      // W will be named "linear/W"
 *      auto W = Variable(linear.WithOpName("W"),
 *                        {2, 2}, DT_FLOAT);
 *      // b will be named "linear/b_3"
 *      int idx = 3;
 *      auto b = Variable(linear.WithOpName("b_", idx),
 *                        {2}, DT_FLOAT);
 *      auto x = Const(linear, {...});  // name: "linear/Const"
 *      auto m = MatMul(linear, x, W);  // name: "linear/MatMul"
 *      auto r = BiasAdd(linear, m, b); // name: "linear/BiasAdd"
 * 
 *  Scope lifetime:
 * 
 *  A new scope is created by calling Scope::NewRootScope. This creates some
 *  resources that are shared by all the child scopes that inherit from this
 *  scope, directly or transitively. For instance, a new scope creates a new
 *  Graph object to which operations are added when the new scope or its
 *  children are used by an Op constructor. The new scope also has a Status
 *  object which will be used to indicate errors by Op-constructor functions
 *  called on any child scope. The Op-constructor functions have to check the
 *  scope's status by calling the ok() method before proceeding to construct the
 *  op.
 * 
 *  Thread safety:
 * 
 *  A {@code Scope} object is NOT thread-safe. Threads cannot concurrently call
 *  op-constructor functions on the same {@code Scope} object. */
@Name("tensorflow::Scope") @NoOffset @Properties(inherit = org.tensorflow.internal.c_api.presets.tensorflow.class)
public class TF_Scope extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_Scope(Pointer p) { super(p); }

  public TF_Scope(@Const @ByRef TF_Scope other) { super((Pointer)null); allocate(other); }
  private native void allocate(@Const @ByRef TF_Scope other);
  public native @ByRef @Name("operator =") TF_Scope put(@Const @ByRef TF_Scope other);

  // The following functions are for users making graphs. They return brand new
  // scopes, or scopes derived from an existing scope object.

  /** Return a new scope.
   *  This creates a new graph and all operations constructed in this graph
   *  should use the returned object as the "root" scope. */
  public static native @ByVal TF_Scope NewRootScope();

  /** Return a new scope. Ops created with this scope will have
   *  {@code name/child_scope_name} as the prefix. The actual name will be unique
   *  in the current scope. All other properties are inherited from the current
   *  scope. If {@code child_scope_name} is empty, the {@code /} is elided. */
  public native @ByVal TF_Scope NewSubScope(BytePointer child_scope_name);
  public native @ByVal TF_Scope NewSubScope(String child_scope_name);

  /** Return a new scope. All ops created within the returned scope will have
   *  names of the form {@code name/StrCat(fragments...)[_suffix]} */

  /** Return a new scope. All ops created within the returned scope will have as
   *  control dependencies the union of operations in the control_deps vector
   *  and the control dependencies of the current scope. */
  public native @ByVal TF_Scope WithControlDependencies(
        @StdVector TF_Operation control_deps);
  /** Same as above, but convenient to add control dependency on the operation
   *  producing the control_dep output. */
  public native @ByVal TF_Scope WithControlDependencies(@Const @ByRef TF_Output control_dep);

  /** Return a new scope. All ops created within the returned scope will have no
   *  control dependencies on other operations. */
  public native @ByVal TF_Scope WithNoControlDependencies();

  /** Return a new scope. All ops created within the returned scope will have
   *  the device field set to 'device'. */
  public native @ByVal TF_Scope WithDevice(BytePointer device);
  public native @ByVal TF_Scope WithDevice(String device);

  /** Returns a new scope.  All ops created within the returned scope will have
   *  their assigned device set to {@code assigned_device}. */
  

  /** Returns a new scope.  All ops created within the returned scope will have
   *  their _XlaCluster attribute set to {@code xla_cluster}. */
  

  /** Return a new scope. All ops created within the returned scope will be
   *  co-located on the device where op is placed.
   *  NOTE: This function is intended to be use internal libraries only for
   *  controlling placement of ops on to devices. Public use is not encouraged
   *  because the implementation of device placement is subject to change. */
  
  /** Convenience function for above. */
  
  /** Clear all colocation constraints. */
  

  /** Return a new scope. The op-constructor functions taking the returned scope
   *  as the scope argument will exit as soon as an error is detected, instead
   *  of setting the status on the scope. */
  public native @ByVal TF_Scope ExitOnError();

  /** Return a new scope. All ops created with the new scope will have
   *  kernel_label as the value for their '_kernel' attribute; */
  

  // The following functions are for scope object consumers.

  /** Return a unique name, using default_name if an op name has not been
   *  specified. */
  public native BytePointer GetUniqueNameForOp(BytePointer default_name);
  public native String GetUniqueNameForOp(String default_name);

  /** Update the status on this scope.
   *  Note: The status object is shared between all children of this scope.
   *  If the resulting status is not Status::OK() and exit_on_error_ is set on
   *  this scope, this function exits by calling LOG(FATAL). */
  public native void UpdateStatus(@Const @ByRef TF_Status s);

  // START_SKIP_DOXYGEN

  /** Update the builder with properties accumulated in this scope. Does not set
   *  status(). */
  // TODO(skyewm): NodeBuilder is not part of public API
  public native void UpdateBuilder(NodeBuilder builder);
  // END_SKIP_DOXYGEN

  public native @Cast("bool") boolean ok();

  // TODO(skyewm): Graph is not part of public API
  

  // TODO(skyewm): Graph is not part of public API
  

  public native @ByVal TF_Status status();

  /** If status() is Status::OK(), convert the Graph object stored in this scope
   *  to a GraphDef proto and return Status::OK(). Otherwise, return the error
   *  status as is without performing GraphDef conversion. */
  

  // START_SKIP_DOXYGEN

  /** If status() is Status::OK(), construct a Graph object using {@code opts} as the
   *  GraphConstructorOptions, and return Status::OK if graph construction was
   *  successful. Otherwise, return the error status. */
  // TODO(josh11b, keveman): Make this faster; right now it converts
  // Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
  // edges from the source and to the sink node, resolves back edges
  // by name), and makes sure the resulting graph is valid.
  

  // Calls AddNode() using this scope's ShapeRefiner. This exists in the public
  // API to prevent custom op wrappers from needing access to shape_refiner.h or
  // scope_internal.h.
  // TODO(skyewm): remove this from public API
  

  // Creates a new root scope that causes all DoShapeInference() calls to return
  // Status::OK() (on the returned scope and any subscopes). Used for testing.
  // TODO(skyewm): fix tests that still require this and eventually remove, or
  // at least remove from public API
  
  // END_SKIP_DOXYGEN

  

  // START_SKIP_DOXYGEN
  @Opaque public static class Impl extends Pointer {
      /** Empty constructor. Calls {@code super((Pointer)null)}. */
      public Impl() { super((Pointer)null); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Impl(Pointer p) { super(p); }
  }
  public native Impl impl();
}
