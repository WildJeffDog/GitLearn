import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 加载 MNIST 数据集
(x_train, _), (x_test, _) = mnist.load_data() # 我们只需要图像，不需要标签

# 将像素值归一化到 [0, 1] 并展平图像
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(f"训练数据形状: {x_train.shape}")
print(f"测试数据形状: {x_test.shape}")

# 定义输入形状和潜在维度
input_dim = 784
latent_dim = 32

# --- 编码器 ---
encoder = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(latent_dim, activation="relu", name="bottleneck"), # 瓶颈层
    ],
    name="encoder",
)

# --- 解码器 ---
decoder = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid"), # 输出层
    ],
    name="decoder",
)

# --- 自编码器（编码器 + 解码器） ---
autoencoder = keras.Sequential(
    [
        encoder,
        decoder,
    ],
    name="autoencoder",
)

# 显示模型摘要
encoder.summary()
decoder.summary()
autoencoder.summary()

# 编译自编码器
autoencoder.compile(optimizer='adam', loss='mse') # 使用均方误差损失

# 训练自编码器
epochs = 20
batch_size = 256

history = autoencoder.fit(x_train, x_train, # 输入和目标相同
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(x_test, x_test)) # 在测试集上评估重建效果

# 预测测试集的重建图像
reconstructed_imgs = autoencoder.predict(x_test)

# --- 可视化 ---
n = 10 # 要显示的数字数量
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("原始图像", loc='left', fontsize=12, pad=10)

    # 显示重建图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("重建图像", loc='left', fontsize=12, pad=10)

plt.suptitle("原始与重建 MNIST 数字对比", fontsize=16)
plt.show()