import tensorflow as tf
import numpy as np
import time
import psutil
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none' 
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


def build_vgg16_architecture():
    base = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model

def build_resnet50_architecture():
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model

def benchmark_model(model_path, model_name):
    print(f"\n{'='*40}")
    print(f"Loading and Benchmarking: {model_name}")
    print(f"{'='*40}")
    
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    
    if 'VGG' in model_name:
        model = build_vgg16_architecture()
    else:
        model = build_resnet50_architecture()
        
    model.load_weights(model_path)
    # ----------------------
    
    ram_after = process.memory_info().rss / (1024 * 1024)
    ram_used = ram_after - ram_before
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    total_flops = get_flops(model)
    
    dummy_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    for _ in range(5):
        _ = model.predict(dummy_image, verbose=0)
        
    print("Benchmarking GPU Inference...")
    gpu_times = []
    for _ in range(50):
        start_time = time.time()
        _ = model.predict(dummy_image, verbose=0)
        gpu_times.append(time.time() - start_time)
    avg_gpu_time = np.mean(gpu_times) * 1000 
    
    print("Benchmarking CPU Inference...")
    cpu_times = []
    with tf.device('/CPU:0'):
        for _ in range(50):
            start_time = time.time()
            _ = model.predict(dummy_image, verbose=0)
            cpu_times.append(time.time() - start_time)
    avg_cpu_time = np.mean(cpu_times) * 1000
    
    print(f"\n--- {model_name} Final Hardware Report ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Total FLOPs:      {total_flops:,}")
    print(f"RAM Usage:        {ram_used:.2f} MB")
    print(f"GPU Infer Time:   {avg_gpu_time:.2f} ms per image")
    print(f"CPU Infer Time:   {avg_cpu_time:.2f} ms per image")


vgg_path = r"Performance Evaluation of ResNet-50 and VGG-16 Architectures in Classifying Chili Pepper (Capsicum annuum) Leaf Diseases\Trained Model\vgg16_chili.keras"
resnet_path = r"Performance Evaluation of ResNet-50 and VGG-16 Architectures in Classifying Chili Pepper (Capsicum annuum) Leaf Diseases\Trained Model\resnet50_chili.keras"

benchmark_model(vgg_path, 'VGG-16')
benchmark_model(resnet_path, 'ResNet-50')