import numpy as np
from KS import KS
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from tqdm import trange
import os


import tqdm
from matplotlib.animation import FuncAnimation


def generate_data(
    L=22,
    N=64,
    dt=0.25,
    steps=2000,
    t_end=None,
    diffusion=1,
    save_path=None,
    seed=None,
    name=None,
):

    # Overwiting the steps if t_end is given
    if t_end is not None:
        steps = int(t_end / dt)
        print("Total steps: ", steps)

    if seed is None:
        seed = np.random.randint(0, 1000000)

    random_state = np.random.RandomState(seed)

    xx = np.linspace(0, L, N)
    u = np.cos((2 * np.pi * xx) / L) + 0.1 * np.cos((4 * np.pi * xx) / L)
    u = u.reshape(1, -1)

    # Randomly choose the initial condition (These are close to 0, TODO Study this)
    u = 0.01 * random_state.rand(1, N)

    system = KS(
        L, N, dt, diffusion=diffusion, initial_conditions=u, rs=random_state
    )

    # Initial condition
    # u = 0.01 * np.random.normal(size=N)  # noisy IC
    # remove zonal mean
    # u = u - u.mean()
    # spectral space variable.
    system.xspec[0] = np.fft.rfft(u)

    uu = []
    tt = []

    for _ in trange(steps):
        system.advance()
        u = system.x.squeeze()
        uu.append(u)
        tt.append(_ * dt)

    print("type: ", type(uu), "shape: ", np.array(uu).shape)
    uu = np.array(uu)
    tt = np.array(tt)

    print("type: ", type(uu), "shape: ", uu.shape)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.DataFrame(uu)

        if name is None:
            name = (
                f"KS_L{L}_N{N}_dt{dt}_steps{steps}_diffusion-k{diffusion}.csv"
            )
        else:
            name = name + ".csv"

        df.to_csv(os.path.join(save_path, name), index=False, header=False)

    return xx, tt, uu


def render_video(
    data,
    predictions=None,
    title="",
    xlabel=r"N",
    frames=None,
    save_path=None,
    dt=1,
):
    """
    Render a video of the predictions and the target.

    Args:
        predictions (N x T x D np.array): The predictions of the model.

        val_target (N x T x D np.array) [Optional]: The target values. Real Data.

        title (str): The title of the plots.

        xlabel (str): The label for the x axis.

        frames (int): The number of frames to render. If None, render all of them.

        save_path (str): The path to save the plot. If None, the video is not saved.

    Returns:
        None
    """

    if predictions is not None:
        # reshape the predictions and the target to 2D
        predictions = predictions.reshape(predictions.shape[1], -1)

    # Set the number of frames to the number of predictions
    if frames is None and predictions is not None:
        frames = predictions.shape[0]
    elif frames is None:
        frames = data.shape[0]

    fig, axs = plt.subplots(figsize=[20, 9.6])

    axs.set_ylim(data.min(), data.max())

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)

    (model_plot,) = axs.plot(data[0], label="Original model")

    # also write the time in the plot
    time_text = axs.text(0.02, 0.95, f"t={0}", transform=axs.transAxes)

    if predictions is not None:
        (prediction_plot,) = axs.plot(predictions[0], label="Predicted model")

    axs.legend()

    def update(frame):
        # if i % 10 == 0:
        #     print(f"Frame {i}/{frames}")

        # update the time in the plot
        time_text.set_text(f"t={frame*dt}")

        model_plot.set_ydata(data[frame])
        if predictions is not None:
            prediction_plot.set_ydata(predictions[frame])
            return model_plot, prediction_plot
        return (model_plot, time_text)

    print("Animating...")
    print()

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=int(200 * dt),
        blit=True,
        repeat=False,
    )

    # progress_callback function using tqdm to show the progress of the animation
    if save_path is not None:

        pbar = tqdm.tqdm(total=frames)

        def progress_callback(current_frame, total_frames):
            pbar.update(current_frame - pbar.n)

        ani.save(
            "".join([save_path, "_video.mp4"]),
            writer="ffmpeg",
            progress_callback=progress_callback,
        )

    plt.show()


def main():

    # xx, tt, uu = generate_data(
    #     L=30, N=128, dt=0.25, steps=160000, save_path=True
    # )

    t_end = 40000
    N = 128
    dt = 0.25
    L = 35
    steps = int(t_end / dt)
    diffusion = 1

    for i in range(20):

        name = (
            f"KS_L{L}_N{N}_dt{dt}_steps{steps}_diffusion-k{diffusion}_run{i}"
        )

        os.system(f"mkdir -p data/KS/L{L}_dt{dt}")

        # check if file exists and skip
        if os.path.exists(f"data/KS/L{L}_dt{dt}/{name}.csv"):
            print(f"Skipping {name}")
            continue

        xx, tt, uu = generate_data(
            L=L,
            N=N,
            dt=dt,
            t_end=t_end,
            save_path=f"data/KS/L{L}_dt{dt}",
            name=name,
        )

        # plt.contourf(uu.T, levels=20)
        # cbar = plt.colorbar()
        # plt.show()

        # exit(0)

        # render_video(
        #     uu,
        #     dt=dt,
        #     title="KS",
        #     save_path=f"videos/{name}_video",
        #     frames=1000,
        # )

        # exit(0)
    # print(uu.shape)
    # exit(0)

    # uu = uu.reshape(1, -1, N)

    # print(uu.shape)
    # plt.plot(uu[2000])
    # plt.show()
    # exit(0)


if __name__ == "__main__":
    main()
