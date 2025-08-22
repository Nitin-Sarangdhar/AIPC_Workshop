import openvino as ov
from openvino.runtime import opset8 as ov_ops
import numpy as np
import cv2 # Still good to confirm it's available

print("OpenVINO version:", ov.__version__)
print("OpenCV version:", cv2.__version__)

try:
    # 1. Create OpenVINO Core object
    core = ov.Core()
    print("OpenVINO Core created successfully.")

    # Define input parameters for the model
    # Shape: (Batch, Channels, Height, Width) - common for image models
    input_shape = [1, 3, 224, 224]
    input_type = ov.Type.f32 # Float32 is common for model inputs

    # Create a parameter (input) node for the model
    param = ov_ops.parameter(input_shape, input_type, name="input_tensor")

    # For this simple example, the output is just the input itself
    # In a real model, you'd have operations like convolutions, activations, etc.
    result = ov_ops.result(param, name="output_tensor")

    # Create the actual OpenVINO model
    model = ov.Model([result], [param], "simple_pass_through_model")
    print(f"\nOpenVINO Model created with input: {model.input().get_shape()} and output: {model.output().get_shape()}")

    # 2. Compile the model for the CPU device
    # You can specify other devices like "GPU", "NPU", etc., if available.
    # "CPU" is always available.
    print(f"Compiling model for device: CPU...")
    compiled_model = core.compile_model(model, "CPU")
    print("Model compiled successfully.")

    # Get input and output information from the compiled model
    input_node = compiled_model.input(0) # Get the first (and only) input
    output_node = compiled_model.output(0) # Get the first (and only) output

    print(f"Compiled Model Input Node: {input_node.get_any_name()} with shape {input_node.get_shape()}")
    print(f"Compiled Model Output Node: {output_node.get_any_name()} with shape {output_node.get_shape()}")


    # 3. Prepare dummy input data (a random image-like array)
    # Ensure the data type matches the model's expected input type (f32)
    dummy_input_data = np.random.rand(*input_shape).astype(np.float32)
    print(f"\nDummy input data created with shape: {dummy_input_data.shape} and dtype: {dummy_input_data.dtype}")

    # 4. Create an inference request and perform inference
    print("Performing inference...")
    # The infer method directly executes the model with the given input
    # It expects a dictionary mapping input names/indices to numpy arrays
    results = compiled_model.infer_new_request({input_node.any_name: dummy_input_data})
    print("Inference completed.")

    # 5. Retrieve the output
    output_data = results[output_node.any_name]
    print(f"\nOutput data received with shape: {output_data.shape} and dtype: {output_data.dtype}")

    # Verify that the output is the same as the input (for this pass-through model)
    # Using np.allclose to handle potential floating point differences
    if np.allclose(dummy_input_data, output_data):
        print("Verification: Input data matches output data (as expected for this pass-through model).")
    else:
        print("Verification: Input data DOES NOT match output data. Something is wrong with the pass-through.")

    print("\nActual OpenVINO model successfully built, compiled, and inferred!")

except Exception as e:
    print(f"\nAn error occurred during OpenVINO model test: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for debugging
    print("Please ensure your OpenVINO installation is complete and there are no conflicts.")
