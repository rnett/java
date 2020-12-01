/* Copyright 2019-2021 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.processor.operator;

import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.javadoc.Javadoc;
import com.google.common.base.CaseFormat;
import com.google.common.base.Strings;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterSpec;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import com.squareup.javapoet.TypeVariableName;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Filer;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.Name;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.TypeParameterElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.NoType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic.Kind;
import org.tensorflow.Names;

/**
 * A compile-time Processor that aggregates classes annotated with {@code
 * org.tensorflow.op.annotation.Operator} and generates the {@code Ops} convenience API. Please
 * refer to the {@code Operator} annotation for details about the API generated for each annotated
 * class.
 *
 * <p>Note that this processor can only be invoked once, in a single compilation run that includes
 * all the {@code Operator} annotated source classes. The reason is that the {@code Ops} API is an
 * "aggregating" API, and annotation processing does not permit modifying an already generated
 * class.
 */
public final class OperatorProcessor extends BaseOperatorProcessor<TypeSpec> {

  private static final TypeName T_DEVICE_SPEC = ClassName.get("org.tensorflow", "DeviceSpec");

  @Override
  protected void write(TypeSpec spec) {
    try {
      JavaFile.builder("org.tensorflow.op", spec)
          .addFileComment(LICENSE)
          .addFileComment("\nThis class has been generated, DO NOT EDIT!\n")
          .skipJavaLangImports(true)
          .build()
          .writeTo(filer);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  @Override
  protected TypeSpec buildGroupClass(OpsSpec spec) {
    //System.out.println("Generating " + spec.className + " class");

    MethodSpec.Builder ctorBuilder =
        MethodSpec.constructorBuilder()
            .addParameter(Names.Ops, "ops")
            .addStatement("this.scope = ops.scope()")
            .addStatement("this.ops = ops");

    TypeSpec.Builder builder =
        TypeSpec.classBuilder(spec.className)
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addJavadoc(
                "An API for building {@code $L} operations as {@link $T Op}s\n\n"
                    + "@see {@link $T}\n",
                spec.groupName,
                Names.Op,
                Names.Ops)
            .addMethods(spec.javaMethods());

    MethodSpec.Builder opsBuilder =
        MethodSpec.methodBuilder("ops")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .returns(Names.Ops)
            .addJavadoc("Get the parent {@link " + Names.Ops.simpleName() + "} object.")
            .addStatement("return ops");

    builder.addMethod(opsBuilder.build());

    addGroupFields(builder, ctorBuilder, spec.subGroups, false);

    builder.addMethod(ctorBuilder.build());

    builder.addField(
        FieldSpec.builder(Names.Scope, "scope")
            .addModifiers(Modifier.PRIVATE, Modifier.FINAL)
            .build());

    builder.addField(
        FieldSpec.builder(Names.Ops, "ops").addModifiers(Modifier.PRIVATE, Modifier.FINAL).build());

    return builder.build();
  }

  @Override
  protected TypeSpec buildTopClass(OpsSpec spec) {
    //System.out.println("Generating " + spec.className + " class");

    MethodSpec.Builder ctorBuilder =
        MethodSpec.constructorBuilder()
            .addParameter(Names.Scope, "scope")
            .addModifiers(Modifier.PRIVATE)
            .addStatement("this.scope = scope", Names.Scope);

    TypeSpec.Builder opsBuilder =
        TypeSpec.classBuilder("Ops")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addJavadoc(
                "An API for building operations as {@link $T Op}s\n<p>\n"
                    + "Any operation wrapper found in the classpath properly annotated as an"
                    + "{@link $T @Operator} is exposed\n"
                    + "by this API or one of its subgroup.\n<p>Example usage:\n<pre>{@code\n"
                    + "try (Graph g = new Graph()) {\n"
                    + "  Ops tf = Ops.create(g);\n"
                    + "  // Operations are typed classes with convenience\n"
                    + "  // builders in Ops.\n"
                    + "  Constant<TInt32> three = tf.constant(3);\n"
                    + "  // Single-result operations implement the Operand\n"
                    + "  // interface, so this works too.\n"
                    + "  Operand<TInt32> four = tf.constant(4);\n"
                    + "  // Most builders are found within a group, and accept\n"
                    + "  // Operand types as operands\n"
                    + "  Operand<TInt32> nine = tf.math.add(four, tf.constant(5));\n"
                    + "  // Multi-result operations however offer methods to\n"
                    + "  // select a particular result for use.\n"
                    + "  Operand<TInt32> result = \n"
                    + "      tf.math.add(tf.unique(s, a).y(), b);\n"
                    + "  // Optional attributes\n"
                    + "  tf.linalg.matMul(a, b, MatMul.transposeA(true));\n"
                    + "  // Naming operators\n"
                    + "  tf.withName(\"foo\").constant(5); // name \"foo\"\n"
                    + "  // Names can exist in a hierarchy\n"
                    + "  Ops sub = tf.withSubScope(\"sub\");\n"
                    + "  sub.withName(\"bar\").constant(4); // \"sub/bar\"\n"
                    + "}\n"
                    + "}</pre>\n",
                Names.Op,
                Names.Operator)
            .addMethods(spec.javaMethods());

    addGroupFields(opsBuilder, ctorBuilder, spec.subGroups, true);

    opsBuilder.addMethod(ctorBuilder.build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withSubScope")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(Names.String, "childScopeName")
            .returns(Names.Ops)
            .addStatement("return new $T(scope.withSubScope(childScopeName))", Names.Ops)
            .addJavadoc(
                "Returns an API that builds operations with the provided name prefix.\n"
                    + "\n@see {@link $T#withSubScope(String)}\n",
                Names.Scope)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withName")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(Names.String, "opName")
            .returns(Names.Ops)
            .addStatement("return new Ops(scope.withName(opName))")
            .addJavadoc(
                "Returns an API that uses the provided name for an op.\n\n"
                    + "@see {@link $T#withName(String)}\n",
                Names.Scope)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withDevice")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(Names.DeviceSpec, "deviceSpec")
            .returns(Names.Ops)
            .addStatement("return new Ops(scope.withDevice(deviceSpec))")
            .addJavadoc(
                "Returns an API that places the created operations on the device(s) matching the provided spec.\n\n"
                    + "@see {@link $T#withDevice(DeviceSpec)}\n",
                Names.Scope)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withControlDependencies")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(Names.IterableOp, "controls")
            .returns(Names.Ops)
            .addStatement("return new Ops(scope.withControlDependencies(controls))")
            .addJavadoc(
                "Returns an API that adds operations to the graph with the provided control dependencies.\n\n"
                    + "@see {@link $T#withControlDependencies(Iterable<Op<?>>)}\n",
                Names.Scope)
            .build());

    opsBuilder.addField(
        FieldSpec.builder(Names.Scope, "scope")
            .addModifiers(Modifier.PRIVATE, Modifier.FINAL)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("scope")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .returns(Names.Scope)
            .addStatement("return scope")
            .addJavadoc("Returns the current {@link $T scope} of this API\n", Names.Scope)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("create")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .addParameter(Names.ExecutionEnvironment, "env")
            .returns(Names.Ops)
            .addStatement("return new Ops(env.baseScope())", Names.Scope)
            .addJavadoc(
                "Creates an API for building operations in the provided execution environment\n")
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("create")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(Names.Ops)
            .addStatement("return create($T.getDefault())", Names.EagerSession)
            .addJavadoc(
                "Creates an API for building operations in the default eager execution environment\n\n"
                    + "<p>Invoking this method is equivalent to {@code Ops.create(EagerSession.getDefault())}.\n")
            .build());

    return opsBuilder.build();
  }

  private static void addGroupFields(TypeSpec.Builder classBuilder, MethodSpec.Builder ctorBuilder, List<OpsSpec> groups, boolean isTopClass) {
    groups.forEach(group -> {
      classBuilder.addField(
          FieldSpec.builder(group.className, group.fieldName)
              .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
              .build()
      );
      ctorBuilder.addStatement("$L = new $T(" + (isTopClass ? "this" : "ops") + ")", group.fieldName, group.className).build();
    });
  }
}
