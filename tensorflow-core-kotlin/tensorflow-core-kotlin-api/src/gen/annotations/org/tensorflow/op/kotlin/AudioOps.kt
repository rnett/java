// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//
// This class has been generated, DO NOT EDIT!
//
package org.tensorflow.op.kotlin

import org.tensorflow.Operand
import org.tensorflow.op.Scope
import org.tensorflow.op.audio.AudioSpectrogram
import org.tensorflow.op.audio.DecodeWav
import org.tensorflow.op.audio.EncodeWav
import org.tensorflow.op.audio.Mfcc
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TString

/**
 * An API for building {@code audio} operations as {@link org.tensorflow.op.Op Op}s
 *
 * @see {@link org.tensorflow.op.Ops}
 */
public class AudioOps(
  /**
   * Get the parent {@link KotlinOps} object.
   */
  public val ops: KotlinOps
) {
  public val java: org.tensorflow.op.AudioOps = ops.java.audio

  /**
   * Returns the current {@link Scope scope} of this API
   */
  public val scope: Scope = ops.scope

  public fun audioSpectrogram(
    input: Operand<TFloat32>,
    windowSize: Long,
    stride: Long,
    vararg options: AudioSpectrogram.Options
  ): AudioSpectrogram = java.audioSpectrogram(input, windowSize, stride, *options)

  public fun decodeWav(contents: Operand<TString>, vararg options: DecodeWav.Options): DecodeWav =
      java.decodeWav(contents, *options)

  public fun encodeWav(audio: Operand<TFloat32>, sampleRate: Operand<TInt32>): EncodeWav =
      java.encodeWav(audio, sampleRate)

  public fun mfcc(
    spectrogram: Operand<TFloat32>,
    sampleRate: Operand<TInt32>,
    vararg options: Mfcc.Options
  ): Mfcc = java.mfcc(spectrogram, sampleRate, *options)
}