import tensorrt as trt


def build_engine(onnx_model_path, max_workspace_size=1 << 30):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX model.')
            for error in range(parser.num_errors):
                print(f"{error}: {parser.get_error(error).desc()}")
            exit(1)

    profile1 = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.add_optimization_profile(profile1)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    # config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)

    with open(f"assets/image_caption/checkpoint/model_engine.trt", "wb") as f:
        f.write(engine)


if __name__ == '__main__':
    onnx_model_path1 = 'assets/image_caption/checkpoint/model.onnx'
    build_engine(onnx_model_path1)
