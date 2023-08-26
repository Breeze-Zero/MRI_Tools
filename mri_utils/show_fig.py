import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

def show(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image,cmap='gray')#
def show_coils(data, slice_nums, cmap='gray'):
    fig = plt.figure(figsize=(10, 10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)   
        plt.axis('off')
        
def plot_multiframe_images(datasets, save_path=None, scale_factor=1.0, zoom_coords=None, zoom_factor=2.5, row_spacing=0, col_spacing=0):
    """
    绘制多帧单通道图像。每行代表一个多帧图像，每列代表一帧。

    :param datasets: list of numpy arrays. 每个数组的维度为(frame, h, w).
    :param save_path: Optional[str]. 如果提供，将图像保存到指定路径。
    :param scale_factor: 缩放图像的因子。默认为1.0，表示不缩放。
    :param zoom_coords: (x, y, width, height) 矩形的坐标和大小。
    :param zoom_factor: 放大因子。
    :param row_spacing: 行与行之间的间隔。
    :param col_spacing: 列与列之间的间隔。
    """

    # 获取所有数据的最小值和最大值
    global_vmin = min([np.min(data) for data in datasets])
    global_vmax = max([np.max(data) for data in datasets])

    num_datasets = len(datasets)
    num_frames = datasets[0].shape[0]
    h, w = datasets[0].shape[1], datasets[0].shape[2]

    fig = plt.figure(figsize=(w * num_frames * scale_factor, h * num_datasets * scale_factor))
    gs = gridspec.GridSpec(num_datasets, num_frames, wspace=col_spacing, hspace=row_spacing)

    if zoom_coords:
        x, y, width, height = zoom_coords

    for i, data in enumerate(datasets):
        for j, frame in enumerate(data):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(frame, cmap='gray', aspect='auto', vmin=global_vmin, vmax=global_vmax)  # 设置vmin和vmax
            
            if zoom_coords:
                rect = Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                
                # 创建放大的区域
                axins = ax.inset_axes([x, y, width * zoom_factor, height * zoom_factor])
                axins.imshow(frame[y:y+height, x:x+width], cmap='gray', aspect='auto', vmin=global_vmin, vmax=global_vmax)  # 设置vmin和vmax
                axins.axis('off')

            ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    

    
def plot_multiframe_images_with_errorsmap(outputs, targets, errors, save_path=None, scale_factor=1.0, error_scale=1.0):
    """
    绘制多帧单通道图像。每行有三种图像：输出、目标和误差图。

    :param outputs: list of numpy arrays representing output images.
    :param targets: list of numpy arrays representing target images.
    :param errors: list of numpy arrays representing error maps.
    :param save_path: Optional[str]. 如果提供，将图像保存到指定路径。
    :param scale_factor: 缩放图像的因子。默认为1.0，表示不缩放。
    :param error_scale: 误差图的放大系数。
    """

    # 获取所有数据的最小值和最大值
    global_vmin = min(np.min(outputs), np.min(targets), np.min(errors))
    global_vmax = max(np.max(outputs), np.max(targets), np.max(errors * error_scale))

    num_datasets = len(outputs)
    num_frames = outputs[0].shape[0]
    h, w = outputs[0].shape[1], outputs[0].shape[2]

    fig = plt.figure(figsize=(w * num_frames * 3 * scale_factor, h * num_datasets * scale_factor))
    gs = gridspec.GridSpec(num_datasets, num_frames * 3, wspace=0, hspace=0)

    for i in range(num_datasets):
        for j in range(num_frames):
            ax1 = fig.add_subplot(gs[i, j])
            ax1.imshow(outputs[i][j], cmap='gray', aspect='auto', vmin=global_vmin, vmax=global_vmax)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[i, j + num_frames])
            ax2.imshow(targets[i][j], cmap='gray', aspect='auto', vmin=global_vmin, vmax=global_vmax)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[i, j + 2*num_frames])
            ax3.imshow(errors[i][j] * error_scale, cmap='gray', aspect='auto', vmin=global_vmin, vmax=global_vmax)
            ax3.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def zoom_in_rectangle(img, ax, zoom, rectangle_xy, rectangle_width, rectangle_height, **kwargs):
    """
    Parameters
    ----------
    img: array-like
        The image data.
    ax: Axes
        Axes to place the inset axes.
    zoom: float
        Scaling factor of the data axes. zoom > 1 will enlargen the coordinates (i.e., "zoomed in"),
            while zoom < 1 will shrink the coordinates (i.e., "zoomed out").
    rectangle_xy: (float or int, float or int)
        The anchor point of the rectangle to be zoomed.
    rectangle_width: float or int
        Rectangle to be zoomed width.
    rectangle_height: float or int
        Rectangle to be zoomed height.

    Other Parameters
    ----------------
    cmap: str or Colormap, default 'gray'
        The Colormap instance or registered colormap name used to map scalar data to colors.
    zoomed_inset_loc: int or str, default: 'upper right'
        Location to place the inset axes.
    zoomed_inset_lw: float or None, default 1
        Zoomed inset axes linewidth.
    zoomed_inset_col: float or None, default black
        Zoomed inset axes color.
    mark_inset_loc1: int or str, default is 1
        First location to place line connecting box and inset axes.
    mark_inset_loc2:  int or str, default is 3
        Second location to place line connecting box and inset axes.
    mark_inset_lw: float or None, default None
        Linewidth of lines connecting box and inset axes.
    mark_inset_ec: color or None
        Color of lines connecting box and inset axes.

    """
    axins = zoomed_inset_axes(ax, zoom, loc=kwargs.get("zoomed_inset_loc", 1))

    rect = patches.Rectangle(xy=rectangle_xy, width=rectangle_width, height=rectangle_height)
    x1, x2 = rect.get_x(), rect.get_x() + rect.get_width()
    y1, y2 = rect.get_y(), rect.get_y() + rect.get_height()

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    mark_inset(
        ax,
        axins,
        loc1=kwargs.get("mark_inset_loc1", 1),
        loc2=kwargs.get("mark_inset_loc2", 3),
        lw=kwargs.get("mark_inset_lw", None),
        ec=kwargs.get("mark_inset_ec", "1.0"),
    )

    axins.imshow(
        img,
        cmap=kwargs.get("cmap", "gray"),
        origin="lower",
        vmin=kwargs.get("vmin", None),
        vmax=kwargs.get("vmax", None),
    )

    for axis in ["top", "bottom", "left", "right"]:
        axins.spines[axis].set_linewidth(kwargs.get("zoomed_inset_lw", 1))
        axins.spines[axis].set_color(kwargs.get("zoomed_inset_col", "k"))

    axins.set_xticklabels([])
    axins.set_yticklabels([])
    
    

    
    
def annotate_ssim_psnr(ax, ssim, psnr, position="right", fontsize=20):
    text_str = f"{ssim:.2f}/{psnr:.2f}"
    ha = "right" if position == "right" else "left"
    
    if position == "right":
        x, ha = ax.get_xlim()[1], "right"
    else:
        x, ha = ax.get_xlim()[0], "left"
    
    ax.text(x, 0, text_str, color='white', ha=ha, va="top", fontsize=fontsize, bbox=dict(boxstyle="round", ec="black", fc="gray", alpha=0))
    
def annotate_target(ax, fontsize=20):
    ax.text(ax.get_xlim()[1], 0, "Target", color='white', ha="right", va="top", fontsize=fontsize, bbox=dict(boxstyle="round", ec="black", fc="gray", alpha=0))
    
def plot_image_sets(inputs, outputs, targets, input_metrics, output_metrics, error_cmap='jet', scale_factor=1.0, row_spacing=0.1, col_spacing=0.1, position="right", fontsize=100):
    """
    展示输入、模型输出和错误图。

    :param input_img: 输入图像数组。
    :param targets: 目标图像数组。
    :param outputs: 每个模型的输出图像列表。
    :param error_maps: 每个模型的错误图列表。
    :param cmap: 错误图的颜色映射。
    """
    # 示例数据
    # input_image = np.zeros((100, 100))#np.random.rand(100, 100)
    # output_images1 = [input_image + np.random.randn(100, 100) * 0.05 for _ in range(3)]
    # output_images2 = [input_image + np.random.randn(100, 100) * 0.05 for _ in range(3)]
    # target_image = input_image

    # input_metrics = [(0.8, 30), (0.85, 32)]
    # output_metrics1 = [(0.7, 28), (0.65, 27), (0.75, 29)]
    # output_metrics2 = [(0.72, 29), (0.67, 28), (0.77, 30)]

    # plot_image_sets([input_image, input_image], [output_images1, output_images2], [target_image, target_image], input_metrics, [output_metrics1, output_metrics2], scale_factor=0.1, row_spacing=0.05, col_spacing=0.05)
    
    
    
    num_sets = len(inputs)
    num_outputs = len(outputs[0])

    fig = plt.figure(figsize=(inputs[0].shape[1] * (1 + num_outputs) * scale_factor, 
                               inputs[0].shape[0] * 2 * num_sets * scale_factor))

    outer_grid = gridspec.GridSpec(num_sets, 1, wspace=0, hspace=row_spacing)

    for set_idx in range(num_sets):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1 + num_outputs,
                                                      subplot_spec=outer_grid[set_idx], wspace=col_spacing, hspace=col_spacing)

        # 绘制输入图像
        ax = plt.Subplot(fig, inner_grid[0, 0])
        ax.imshow(inputs[set_idx], cmap='gray', vmin=np.min(targets[set_idx]), vmax=np.max(targets[set_idx]))
        ax.axis('off')
        annotate_ssim_psnr(ax, *input_metrics[set_idx])
        fig.add_subplot(ax)

        # 绘制输出图像
        for output_idx in range(num_outputs):
            ax = plt.Subplot(fig, inner_grid[0, output_idx + 1])
            ax.imshow(outputs[set_idx][output_idx], cmap='gray', vmin=np.min(targets[set_idx]), vmax=np.max(targets[set_idx]))
            ax.axis('off')
            annotate_ssim_psnr(ax, *output_metrics[set_idx][output_idx], position=position, fontsize=fontsize)
            fig.add_subplot(ax)

        # 绘制目标图像
        ax = plt.Subplot(fig, inner_grid[1, 0])
        ax.imshow(targets[set_idx], cmap='gray')
        ax.axis('off')
        annotate_target(ax, fontsize=fontsize)
        fig.add_subplot(ax)

        # 计算并绘制误差图
        for output_idx in range(num_outputs):
            error_map = np.abs(outputs[set_idx][output_idx] - targets[set_idx])
            ax = plt.Subplot(fig, inner_grid[1, output_idx + 1])
            im = ax.imshow(error_map, vmin=0, vmax=np.max(targets[set_idx])/5, cmap=error_cmap)
            ax.axis('off')
            fig.add_subplot(ax)

    # 添加一个全局误差条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    # plt.tight_layout()
    plt.show()

