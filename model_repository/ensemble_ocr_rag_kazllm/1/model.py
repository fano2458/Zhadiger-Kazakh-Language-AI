import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            response_sender = request.get_response_sender()
            texts = pb_utils.get_input_tensor_by_name(request, "texts")
            user_request = pb_utils.get_input_tensor_by_name(request, "user_request")
            file_paths = pb_utils.get_input_tensor_by_name(request, "file_paths")
            task = pb_utils.get_input_tensor_by_name(request, "task")

            infer_request = pb_utils.InferenceRequest(
                model_name="ocr",
                inputs=[texts],
                output_names=["output"]
            )

            infer_response = infer_request.execute() # exec 

            ocr_output = infer_response.output_tensors()

            # send to rubert model
            infer_request = pb_utils.InferenceRequest(
                model_name="rag",
                inputs=[ocr_output, user_request, file_paths],
                output_names=["output"]
            )

            infer_response = infer_request.execute() # exec

            rag_output = infer_response.output_tensors()

            # send to kazllm model - using decoupled mode
            infer_request = pb_utils.InferenceRequest(
                model_name="kazllm",
                inputs=[rag_output, task, user_request],
                requested_output_names=["output"]
            )

            # For decoupled mode, we need to handle streaming responses
            def callback(result):
                if result.has_error():
                    # Handle error
                    return
                
                # Forward the streaming result to the client
                for output_tensor in result.output_tensors():
                    inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                    response_sender.send(inference_response)
                
                # Check if this is the last response
                if result.is_last_response():
                    # Complete the response stream
                    response_sender.send(
                        pb_utils.InferenceResponse(output_tensors=[]),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            # Execute asynchronously to handle streaming
            infer_request.async_exec(callback=callback)

            # No response to append as we're handling it via the callback
            # and response_sender in decoupled mode

        # Return None since we're using decoupled mode
        return None