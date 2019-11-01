import torch


def infovae_loss_fn(w_infovae, z_dim, device, **kwargs):
    """
    InfoVAE: Information Maximizing Variational Autoencoders]
    by Shengjia Zhao, Jiaming Song, Stefano Ermon
    https://arxiv.org/abs/1706.02262
    :param w_infovae:
    :param z_dim:
    :param device:
    :param kwargs:
    :return:
    """
    z = kwargs['z']
    z_true = torch.randn(1000, z_dim).to(device)
    return compute_mmd(z_true, z) * w_infovae


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def compute_kernel(x, y):
    # x_size = tf.shape(x)[0]
    # y_size = tf.shape(y)[0]
    # dim = tf.shape(x)[1]
    # tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    # tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    # return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view([x_size, 1, dim]).float().repeat([1, y_size, 1])
    tiled_y = y.view([1, y_size, dim]).float().repeat([x_size, 1, 1])
    # print(tiled_x)
    # print(tiled_y)
    # print(dim)
    # t = tiled_x - tiled_y
    return torch.exp(-torch.mean((tiled_x - tiled_y).float() ** 2, dim=2) / dim)
