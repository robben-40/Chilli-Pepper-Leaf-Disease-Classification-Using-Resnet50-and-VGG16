import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def build_resnet50():
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    preprocess_func = tf.keras.applications.resnet50.preprocess_input
    
    return model, preprocess_func

CLASS_NAMES = [
    'Bacterial Spot', 
    'Cercospora Leaf Spot', 
    'Curl Virus', 
    'Healthy Leaf', 
    'Nutrition Deficiency', 
    'White Spot'
]

def predict_and_visualize(image_path, model_path):
    print("Loading ResNet-50 model... please wait.")
    
    model, preprocess_func = build_resnet50()
    model.load_weights(model_path)
    

    img_pil = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img_pil) 
    

    img_preprocessed = preprocess_func(np.expand_dims(img_array, axis=0))
    

    predictions = model.predict(img_preprocessed)[0]
    class_idx = np.argmax(predictions)
    

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Chili Leaf Disease Diagnosis (ResNet-50)', fontsize=22, fontweight='bold')

    img_display = tf.keras.utils.load_img(image_path) 
    ax1.imshow(img_display)
    ax1.set_title(f'Test Image: {os.path.basename(image_path)}', fontsize=14, pad=10)
    ax1.axis('off') 
    
    y_pos = np.arange(len(CLASS_NAMES))
    ax2.barh(y_pos, predictions, color='#2E8B57', align='center', alpha=0.9) 
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ax2.invert_yaxis() 
    ax2.set_xlabel('ResNet Model Confidence Score (%)', fontsize=14)
    ax2.set_xlim(0, 1.0) 
    
    ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=12)
    

    ax2.get_yticklabels()[class_idx].set_color('red')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE = r'Performance Evaluation of ResNet-50 and VGG-16 Architectures in Classifying Chili Pepper (Capsicum annuum) Leaf Diseases\test_image.jpg' 
    MODEL_PATH = r'Performance Evaluation of ResNet-50 and VGG-16 Architectures in Classifying Chili Pepper (Capsicum annuum) Leaf Diseases\Trained Model\resnet50_chili.keras' 
    
    if os.path.exists(TEST_IMAGE):
        predict_and_visualize(TEST_IMAGE, MODEL_PATH)
    else:
        print(f"Error: Could not open the image '{TEST_IMAGE}'.")