

#include <math.h>
#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

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
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(10, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);


  std::vector<std::vector<float>> content;
std::vector<float> row;
std::vector<float> pred;
std::string line, word;


// Here I have to manually enter the test case
 //std::vector<float> arr = {3.0,  0.0, 1.0,   0.0,  0.0,  0.0,     0.0,     3.0,       3.0,      1.0};

    std::fstream file1("tensorflow/lite/micro/examples/hello_world/pred.csv", std::ios::in);
    if(file1.is_open())
    {
        getline(file1,line);
    while(getline(file1, line))
    {
    
    std::stringstream str(line);
    getline(str,word,',');
    while(getline(str, word, ','))
    pred.push_back(std::stof(word));
    }
    }
    else
    std::cout<<"Could not open the file\n";

    std::fstream file2("tensorflow/lite/micro/examples/hello_world/x_test.csv", std::ios::in);
    if(file2.is_open())
    {
        getline(file2,line);
    while(getline(file2, line))
    {
    row.clear();
    
    std::stringstream str(line);
    getline(str,word,',');
    while(getline(str, word, ','))
    row.push_back(std::stof(word));
    content.push_back(row);
    }
    }
    else
    std::cout<<"Could not open the file\n";

     int cnt=0;

    for(int j=0;j<(int)pred.size();j++){
          
    float input_survived_data[10];
    for(int k=0;k<10;k++){
         input_survived_data[k] = content[j][k];
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

   int x1 = (double)pred[j]>0.5?1:0;
   int x2 = (double)output->data.f[0]>0.5?1:0;

   if(x1==x2) cnt++;


    std::cout <<std::endl<< "pred: ";
    std::cout<<x1<<std::endl;
    std::cout <<std::endl<< "output: ";
    std::cout << x2 << std::endl;
    //std::cout << 
    std::cout << std::endl;

      TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
      TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
      TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);





    } //





 

std :: cout << "cnt :" << cnt << " " << (int)pred.size() <<(int)content.size();


  
  

}

TF_LITE_MICRO_TESTS_END


