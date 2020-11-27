

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/lib/core/status.h"

namespace{
class JavaTapeTensor {
 public:
  JavaTapeTensor(tensorflow::int64 id, tensorflow::DataType dtype,
               const tensorflow::TensorShape& shape)
      : id_(id), dtype_(dtype), shape_(shape) {}
//  JavaTapeTensor(tensorflow::int64 id, tensorflow::DataType dtype,
//               PyObject* shape)
//      : id_(id), dtype_(dtype), shape_(shape) {
//    Py_INCREF(absl::get<1>(shape_));
//  }
  JavaTapeTensor(const JavaTapeTensor& other) {
    id_ = other.id_;
    dtype_ = other.dtype_;
    shape_ = other.shape_;
  }

  ~JavaTapeTensor() {
  }
  tensorflow::int64 GetID() const { return id_; }
  tensorflow::DataType GetDType() const { return dtype_; }

//  PyObject* OnesLike() const;
//  PyObject* ZerosLike() const;

 private:
  tensorflow::int64 id_;
  tensorflow::DataType dtype_;
  tensorflow::TensorShape shape_;

  // Note that if shape_.index() == 1, meaning shape_ contains a PyObject, that
  // PyObject is the tensor itself. This is used to support tf.shape(tensor) for
  // partially-defined shapes and tf.zeros_like(tensor) for variant-dtype
  // tensors.
//  absl::variant<tensorflow::TensorShape, PyObject*> shape_;
};

//static JavaTapeTensor TapeTensorFromTensor(PyObject* tensor);
}