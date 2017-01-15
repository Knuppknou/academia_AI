"""Module images.py with functions to plot two-dimensional pixel maps."""


def plot_image(image, ax=None, title=None, figsize=(3, 3),
               vmin=0, vmax=1, cmap='Greys'):
    '''Docstring.'''

    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = plt.gca()
    imgplot = ax.imshow(image, aspect='equal', cmap=cmap,
                        interpolation='none', vmin=vmin, vmax=vmax)
    if cmap != 'Greys':
        plt.colorbar(imgplot, ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    return ax

def plot_images(
        images,
        ax=None,
        title=None,
        base_size=(
            3,
            3),
    vmin=0,
    vmax=1,
        cmap='Greys'):
    ''' Docstring. '''
    # Check if user really provided 3-dimensional numpy object
    if len(images.shape) < 3:
        return plot_image(
            images,
            title=title,
            figsize=base_size,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)
    if len(images.shape) > 3:
        print('Error: Dimensions of data to plot not understood! ')
        return 0

    # Number of images to plot (along depth axis z)
    n = images.shape[0]
    plots_per_row = 3
    if n <= plots_per_row:
        f, axs = plt.subplots(1, n, figsize=(base_size[0] * n, base_size[1]))
        if n == 1:
            return plot_image(
                images[0],
                ax=axs,
                title='z = 0',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap)
        for z in range(n):
            current_image = images[z]
            plot_image(
                current_image,
                ax=axs[z],
                title='z = ' +
                str(z),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap)
    else:
        rows = (n - 1) // plots_per_row + 1
        f, axs = plt.subplots(rows, plots_per_row, figsize=(
            base_size[0] * rows, base_size[1] * plots_per_row))
        for z in range(n):
            current_image = images[z]
            plot_image(
                current_image,
                ax=axs[
                    z //
                    plots_per_row][
                    z %
                    plots_per_row],
                title='z = ' +
                str(z),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap)
        for z in range(n, rows * plots_per_row):
            axs[z // plots_per_row][z % plots_per_row].axis('off')
    return None
