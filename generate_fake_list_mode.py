import numpy as np
import parallelproj
from functools import partial


def convert_to_list_mode(xstart, xend, mean_data, scale_factor=100):
    # ravel detector start and end points
    x_start = xstart.reshape(-1, 3)
    x_end = xend.reshape(-1, 3)
    # ravel image
    continuous_data = mean_data.ravel()
    # scale data
    scale = 1 / scale_factor
    # contaminate data
    data_int = np.random.poisson(continuous_data * scale)
    # remove zeros
    list_mode_lors = [i for i, elem in enumerate(data_int) if elem != 0]
    non_zero_data_int = [data_int[i] for i in list_mode_lors]
    # create list of lor indexes
    lor_idxs = [[list_mode_lors[i]] * elem for i, elem in enumerate(non_zero_data_int)]
    lor_idxs = [item for sublist in lor_idxs for item in sublist]
    # shuffle
    np.random.shuffle(np.asarray(lor_idxs))
    # create look-up for detectors
    idx_coord_lookup = np.concatenate((x_start, x_end), axis=0)
    # create list of detectors
    det_1 = lor_idxs
    det_2 = [lor_idx + x_start.shape[0] - 1 for lor_idx in lor_idxs]
    # stack
    dets = np.stack((det_1, det_2), axis=0)
    # shuffle
    np.random.shuffle(dets)
    lm = dets
    print(f"Number of counts: {lm.shape[1]}, scale factor: {scale_factor}")
    return lm, idx_coord_lookup


def get_fwd_bwd(image_origin, voxel_size, image_shape):
    partial_fwd = partial(
        parallelproj.joseph3d_fwd, img_origin=image_origin, voxsize=voxel_size
    )
    partial_bwd = partial(
        parallelproj.joseph3d_back,
        img_shape=image_shape,
        img_origin=image_origin,
        voxsize=voxel_size,
    )
    return partial_fwd, partial_bwd


if __name__ == "__main__":
    print("TAKEN FROM GEORG'S REPO!")

    import array_api_compat.numpy as xp
    import parallelproj
    import matplotlib.pyplot as plt
    from array_api_compat import to_device

    dev = "cpu"
    image_shape = (128, 128)
    voxel_size = (2.0, 2.0)
    image_origin = (-127.0, -127.0)
    radial_positions = to_device(xp.linspace(-128, 128, 200), dev)
    view_angles = to_device(xp.linspace(0, xp.pi, 180, endpoint=False), dev)
    radius = 200.0
    proj2d = parallelproj.ParallelViewProjector2D(
        image_shape, radial_positions, view_angles, radius, image_origin, voxel_size
    )
    img = to_device(xp.zeros(proj2d.in_shape, dtype=xp.float32), dev)
    img[32:96, 32:64] = 1.0
    img_gt = img.copy()
    img_fwd = proj2d(img)
    img_fwd_back = proj2d.adjoint(img_fwd)
    x_start = proj2d._xstart.reshape(-1, 3)
    x_end = proj2d._xend.reshape(-1, 3)
    mean_data = img_fwd.ravel()
    lm, idx_coord_lookup = convert_to_list_mode(x_start, x_end, mean_data)
    image_shape = (1, 128, 128)
    voxel_size = (1.0, 2.0, 2.0)
    image_origin = (0.0, -127.0, -127.0)
    fwd, bwd = get_fwd_bwd(image_origin, voxel_size, image_shape)
    # create projector
    img = np.ones(image_shape)
    # MLEM
    start_detector = idx_coord_lookup[lm[0]]
    end_detector = idx_coord_lookup[lm[1]]
    for i in range(10):
        fwd_proj = fwd(start_detector, end_detector, img)
        img = img * parallelproj.joseph3d_back(
            start_detector,
            end_detector,
            image_shape,
            image_origin,
            voxel_size,
            1 / fwd_proj,
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.colorbar(ax[0].imshow(img_gt[:, :]), ax=ax[0])
    ax[0].axis("off")
    ax[0].set_title("Ground truth")
    fig.colorbar(ax[1].imshow(img[0, :, :]), ax=ax[1])
    ax[1].axis("off")
    ax[1].set_title("MLEM")
    plt.savefig("mlem.png")
    plt.close()
