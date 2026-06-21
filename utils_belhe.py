import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def export_field(query_func, xmin, xmax, ymin, ymax, resolution=64, dim=1, device='cpu') -> np.ndarray:
    N = 64
    X = torch.linspace(xmin, xmax, resolution).split(N)
    Y = torch.linspace(ymin, ymax, resolution).split(N)

    u = np.empty([resolution, resolution, dim])
    with torch.no_grad():
        for yi, ys in enumerate(Y):
            for xi, xs in enumerate(X):
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                xx = xx.reshape(-1, 1)
                yy = yy.reshape(-1, 1)
                pts = torch.cat([yy, xx], dim=1).to(device) # Concatenate in (y, x) order to mimic the convention of DANN
                val = query_func(pts)
                val = val.reshape(len(ys), len(xs), dim).detach().cpu().numpy()
                u[yi*N:yi*N+len(ys), xi*N:xi*N+len(xs), :] = val
    return u

def visualise_output(query_field, query_gt, xmin, xmax, ymin, ymax, vmin, vmax, levels=None, resolution=64):
    bbox = (xmin, xmax, ymin, ymax)
    # Generate Ground truth field for white dashed reference isolevel lines
    field_gt = export_field(
        query_gt,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        resolution=resolution,
        device=torch.device("cuda"),
    ).squeeze(-1)
    # Generate the network output
    field_out = export_field(
        query_field,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        resolution=resolution,
        device=torch.device("cuda"),
    ).squeeze(-1)
    # Set up figure
    figure = plt.figure()
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()

    # Background: Network output
    ax.imshow(field_out, origin="lower", cmap="winter", zorder=0, vmin=vmin, vmax=vmax, extent=bbox)
    # Layer 1: GT isolevel
    ax.contour(field_gt, levels=levels, colors="white", linestyles="dashed", linewidths=1.3, origin="lower", zorder=1, vmin=vmin, vmax=vmax, extent=bbox)
    # Layer 2: Network isolevel
    ax.contour(field_out, levels=levels, colors="orange", linestyles="solid", linewidths=1.3, origin="lower", zorder=2, vmin=vmin, vmax=vmax, extent=bbox)

    return figure

def write_height_mesh(fn: os.PathLike, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    assert x.shape == z.shape and y.shape == z.shape, "x, y, z must have the same shape, use meshgrid"
    height, width = z.shape

    with open(fn, "w+b") as f:
        f.write("ply\n".encode("ascii"))
        f.write("format binary_little_endian 1.0\n".encode("ascii"))
        f.write(f"element vertex {z.size}\n".encode("ascii"))
        f.write("property float x\n".encode("ascii"))
        f.write("property float y\n".encode("ascii"))
        f.write("property float z\n".encode("ascii"))
        f.write(f"element face {(height-1)*(width-1)}\n".encode("ascii"))
        f.write("property list uchar int vertex_index\n".encode("ascii"))
        f.write("end_header\n".encode("ascii"))

        # Write vertices
        for yi in range(height):
            for xi in range(width):
                f.write(np.float32(x[yi, xi]).astype('<f4').tobytes())
                f.write(np.float32(y[yi, xi]).astype('<f4').tobytes())
                f.write(np.float32(z[yi, xi]).astype('<f4').tobytes())

        # Write faces
        for yi in range(height-1):
            for xi in range(width-1):
                f.write(np.uint8(4).astype('<u1').tobytes())
                f.write(np.int32(yi*width + xi).astype('<i4').tobytes())
                f.write(np.int32(yi*width + (xi+1)).astype('<i4').tobytes())
                f.write(np.int32((yi+1)*width + (xi+1)).astype('<i4').tobytes())
                f.write(np.int32((yi+1)*width + xi).astype('<i4').tobytes())
