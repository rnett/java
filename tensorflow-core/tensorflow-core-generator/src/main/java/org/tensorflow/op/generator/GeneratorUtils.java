/*
  Copyright 2021 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================
 */
package org.tensorflow.op.generator;

import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.commonmark.node.Node;
import org.commonmark.parser.Parser;
import org.tensorflow.op.generator.javadoc.JavaDocRenderer;
import org.tensorflow.proto.framework.OpDef.ArgDef;

/**
 * Utilities for op generation
 */
final class GeneratorUtils {

  private static final Parser parser = Parser.builder().build();

  /**
   * Convert a Python style name to a Java style name.
   *
   * Does snake_case -> camelCase and handles keywords.
   *
   * Not valid for class names, meant for fields and methods.
   *
   * Generally you should use {@link ClassGenerator#getJavaName(ArgDef)}.
   */
  static String javaizeMemberName(String name) {
    StringBuilder result = new StringBuilder();
    boolean capNext = Character.isUpperCase(name.charAt(0));
    for (char c : name.toCharArray()) {
      if (c == '_') {
        capNext = true;
      } else if (capNext) {
        result.append(Character.toUpperCase(c));
        capNext = false;
      } else {
        result.append(c);
      }
    }
    name = result.toString();
    switch (name) {
      case "size":
        return "sizeOutput";
      case "if":
        return "ifOutput";
      case "while":
        return "whileOutput";
      case "for":
        return "forOutput";
      case "case":
        return "caseOutput";
      default:
        return name;
    }
  }

  /**
   * Convert markdown descriptions to JavaDocs.
   */
  static String parseDocumentation(String docs) {
    Node document = parser.parse(docs);
    JavaDocRenderer renderer = JavaDocRenderer.builder().build();
    return renderer.render(document);
  }


}
