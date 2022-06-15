

#include <math.h>
#include <bits/stdc++.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/hello_world_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Define the input and the expected output
 
  tflite::MicroErrorReporter micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

  constexpr int kTensorArenaSize = 10*2000;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
 
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(10, input->dims->data[1]);
  // The input is an 8 bit integer value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  // Get the input quantization parameters
 std::vector<float> arr = {3.0,  0.0,     1.0,   0.0,  0.0,  0.0,     0.0,     3.0,       3.0,      1.0};


  
      float input_survived_data[10];
    for(int k=0;k<10;k++){
         input_survived_data[k] = arr[k];
    }
 
  TF_LITE_REPORT_ERROR(&micro_error_reporter, "%d", input->bytes);
  int len = input->bytes/sizeof(float);
  std:: cout << "len " << len << std::endl;
    for (int i = 0; i < len; ++i) {
        input->data.f[i] = input_survived_data[i];
    }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed on some x");
  }

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);


 TfLiteTensor* output = interpreter.output(0);
std::cout <<std::endl<< "output: ";
std::cout << output->data.f[0] << " " << output->data.f[1]<< std::endl;
//std::cout << 
std::cout << std::endl;

TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

   


  



}

TF_LITE_MICRO_TESTS_END
