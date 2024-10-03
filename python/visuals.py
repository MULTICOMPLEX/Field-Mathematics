from matplotlib import widgets
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from constants import *
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb
import pyfftw
from functions import *
from matplotlib.widgets import Slider
from matplotlib.widgets import Button


def plot_spectrum(Ψ, title = 'title'):
    fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
    ax = fig.gca()
    set_axis_color(ax)
    Fs = len(Ψ) # Example value, set this to the actual sampling rate of your signal
    s, f = mlab.psd(Ψ, NFFT= Fs, Fs = Fs)
    plt.loglog(f, s)
    plt.grid(True, which='both', alpha = 0.4)
    plt.xlabel('Frequency (Hz)')
    #plt.ylabel('PSD (Unit**2/Hz)')
    plt.ylabel('PSD (a.u.)')
    plt.title(title, color='white')
    plt.grid(True)

px = 1 / plt.rcParams['figure.dpi']

viridis = plt.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
mc = np.array([0, 43/256, 54/256, 1])

newcolors[:150, :] = mc
newcmp = ListedColormap(newcolors)

def complex_to_rgb(Z: np.ndarray):
    """Convert complex values to their rgb equivalent.
    Parameters
    ----------
    Z : array_like
        The complex values.
    Returns
    -------
    array_like
        The rgb values.
    """
    # using HSV space
    r = np.abs(Z)
    arg = np.angle(Z)

    h = (arg + np.pi) / (2 * np.pi)
    s = np.ones(h.shape)
    v = r / np.amax(r)  # alpha

    c = hsv_to_rgb(np.moveaxis(np.array([h, s, v]), 0, -1))  # --> tuple

    # Identify the black pixels and replace with red pixels
    # black_pixels = (img_array[:, :, 0] < 0.3) & (img_array[:, :, 1] < 0.3) & (img_array[:, :, 2] < 0.3)
    # img_array[np.where(black_pixels)] = k

    return c


def complex_to_rgba(Z: np.ndarray, max_val: float = 1.0):
    r = np.abs(Z)
    arg = np.angle(Z)

    h = (arg + np.pi) / (2 * np.pi)
    s = np.ones(h.shape)
    v = np.ones(h.shape)  # alpha
    rgb = hsv_to_rgb(np.moveaxis(np.array([h, s, v]), 0, -1))  # --> tuple

    abs_z = np.abs(Z) / max_val
    abs_z = np.where(abs_z > 1., 1., abs_z)
    return np.concatenate((rgb, abs_z.reshape((*abs_z.shape, 1))), axis=(abs_z.ndim))


def complex_plot_2D(
    Ψ_plot=np.ndarray,
    extent=10.,
    V=np.ndarray,
    Vmin=-10.,
    Vmax=10.,
    t=0,
    xlim=None,
    ylim=None,
    figsize=(640*px, 640*px),
    potential_saturation=0.5,
    wavefunction_saturation=0.5,
    title = ""
):
    fig = plt.figure(figsize=figsize, facecolor='#002b36')
    
    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    ax.set_xlabel("[Å]")
    ax.set_ylabel("[Å]")
    ax.set_title(title, color='white')

    if(t):
        time_ax = ax.text(0.97, 0.97, "",  color="white",
                      transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format(
        "%.3f" % (t/const["femtoseconds"])))

    if xlim != None:
        ax.set_xlim(np.array(xlim)/const["femtoseconds"])
    if ylim != None:
        ax.set_ylim(np.array(ylim)/const["femtoseconds"])

    L = extent/const["femtoseconds"]

    ax.imshow((V + Vmin)/(Vmax-Vmin),
              vmax=1.0/potential_saturation, vmin=0, cmap=newcmp, origin="lower",
              interpolation="gaussian", extent=[-L/2, L/2, -L/2, L/2])
     
    k = np.amax(np.abs(Ψ_plot))
                  
    ax.imshow(complex_to_rgba(Ψ_plot, max_val=k), origin="lower",
              interpolation="gaussian", extent=[-L/2, L/2, -L/2, L/2])

    plt.show()


def complex_plot(x, phi):
    plt.plot(x, np.abs(phi), label='$|\psi(x)|$')
    plt.plot(x, np.real(phi), label='$Re|\psi(x)|$')
    plt.plot(x, np.imag(phi), label='$Im|\psi(x)|$')
    plt.legend(loc='lower left')
    plt.show()
    return


class ComplexSliderWidget(widgets.AxesWidget):
    """
    A circular complex slider widget for manipulating complex
    values.

    References:
    - https://matplotlib.org/stable/api/widgets_api.
    - https://github.com/matplotlib/matplotlib/blob/
    1ba3ff1c273bf97a65e19892b23715d19c608ae5/lib/matplotlib/widgets.py
    """

    def __init__(self, ax, angle, r, animated=False):
        line, = ax.plot([angle, angle], [0.0, r], linewidth=2.0)
        super().__init__(ax)
        self._rotator = line
        self._is_click = False
        self.animated = animated
        self.update = lambda x, y: None
        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)

    def get_artist(self):
        return self._rotator

    def _click(self, event):
        self._is_click = True
        self._update_plots(event)

    def _release(self, event):
        self._is_click = False

    def on_changed(self, update):
        self.update = update

    def _motion(self, event):
        self._update_plots(event)

    def _update_plots(self, event):
        if (self._is_click and event.xdata != None
                    and event.ydata != None
                    and event.x >= self.ax.bbox.xmin and
                    event.x < self.ax.bbox.xmax and
                    event.y >= self.ax.bbox.ymin and
                    event.y < self.ax.bbox.ymax
                ):
            phi, r = event.xdata, event.ydata
            if r < 0.2:
                r = 0.0
            self.update(phi, r)
            self._rotator.set_xdata([phi, phi])
            self._rotator.set_ydata([0.0, r])
            if not self.animated:
                event.canvas.draw()


def superpositions_1D(eigenstates, states, energies, extent, fps=30, total_time=20, **kw):
    """
    Visualize the time evolution of a superposition of energy eigenstates.
    The circle widgets control the relative phase of each of the eigenstates.
    These widgets are inspired by the circular phasors from the
    quantum mechanics applets by Paul Falstad:
    https://www.falstad.com/qm1d/
    """

    total_frames = fps * total_time

    coeffs = None

    def get_norm_factor(psi): return 1.0 / \
        (np.sqrt(np.sum(psi*np.conj(psi)))+1e-6)
    animation_data = {'ticks': 0, 'norm': get_norm_factor(eigenstates[0]),
                      'is_paused': False}
    psi0 = eigenstates[0]*get_norm_factor(eigenstates[0])
    if isinstance(states, int) or isinstance(states, float):
        coeffs = np.array([1.0 if i == 0 else 0.0 for i in range(states)],
                          dtype=np.complex128)
        eigenstates = eigenstates[0: states]
    else:
        coeffs = states
        eigenstates = eigenstates[0: len(states)]
        states = len(states)
        psi0 = np.dot(coeffs, eigenstates)
        animation_data['norm'] = get_norm_factor(psi0)
        psi0 *= animation_data['norm']

    params = {'dt': 0.001,
              'xlim': [-extent/2.0,
                       extent/2.0],
              'save_animation': False,
              'frames': 120
              }
    for k in kw.keys():
        params[k] = kw[k]

    fig = plt.figure(figsize=(16/9 * 5.804 * 0.9, 5.804), facecolor='#002b36')
    grid = plt.GridSpec(5, states)
    ax = fig.add_subplot(grid[0:3, 0:states])

    ax.set_facecolor('#002b36')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.set_title("Superpositions", color='white')

    ax.set_xlabel("[Å]")
    x = np.linspace(-extent/2.0,
                    extent/2.0,
                    len(eigenstates[0]))
    ax.set_yticks([])
    ax.set_xlim(np.array(params['xlim'])/Å)

    line1, = ax.plot(x/Å, np.real(eigenstates[0]), label='$Re|\psi(x)|$')
    line2, = ax.plot(x/Å, np.imag(eigenstates[0]), label='$Im|\psi(x)|$')
    line3, = ax.plot(x/Å, np.abs(eigenstates[0]), label='$|\psi(x)|$')
    ax.set_ylim(-1.7*np.amax(np.abs(psi0)), 1.7*np.amax(np.abs(psi0)))

    leg = ax.legend(facecolor='#002b36', loc='lower left')
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())

    def make_update(n):
        def update(phi, r):
            animation_data['is_paused'] = True
            coeffs[n] = r*np.exp(1.0j*phi)
            psi = np.dot(coeffs, eigenstates)
            animation_data['norm'] = get_norm_factor(psi)
            line1.set_ydata(np.real(psi))
            line2.set_ydata(np.imag(psi))
            line3.set_ydata(np.abs(psi))
        return update

    widgets = []
    circle_artists = []
    for i in range(states):
        circle_ax = fig.add_subplot(grid[4, i], projection='polar')

        plt.setp(circle_ax.spines.values(), color='white')

        circle_ax.set_facecolor('#002b36')
        circle_ax.set_title(str(i)  # + '\nE=' + str() + '$E_0$'
                            , color='white')
        circle_ax.set_xticks([])
        circle_ax.set_yticks([])

        widgets.append(ComplexSliderWidget(circle_ax, 0.0, 1.0, animated=True))
        widgets[i].on_changed(make_update(i))
        circle_artists.append(widgets[i].get_artist())
    artists = circle_artists + [line1, line2, line3]

    def func(*args):
        animation_data['ticks'] += 1
        e = 1.0
        if animation_data['is_paused']:
            animation_data['is_paused'] = False
        else:
            e *= np.exp(-1.0j*energies[0:states]*params['dt'])
        np.copyto(coeffs, coeffs*e)
        norm_factor = animation_data['norm']
        psi = np.dot(coeffs*norm_factor, eigenstates)
        line1.set_ydata(np.real(psi))
        line2.set_ydata(np.imag(psi))
        line3.set_ydata(np.abs(psi))
        if animation_data['ticks'] % 2:
            return [line1, line2, line3]
        else:
            for i, c in enumerate(coeffs):
                phi, r = np.angle(c), np.abs(c)
                artists[i].set_xdata([phi, phi])
                artists[i].set_ydata([0.0, r])
            return artists
    ani = animation.FuncAnimation(fig, func, blit=True, interval=1000.0/60.0,
                                  frames=None if (not params['save_animation']) else
                                  total_frames)
    if params['save_animation'] == True:
        ani.save('superpositions.gif', fps=fps, metadata=dict(artist='Me'))
        return
    else:
        plt.show()


def animate_1D(
        Ψ=np.ndarray,
        X=np.ndarray,
        V=np.ndarray,
        Vmin=-10,
        Vmax=10,
        xlim=None,
        ylim=None,
        figsize=(16/9 * 5.804 * 0.9, 5.804),
        animation_duration=5,
        fps=20,
        total_time=10,
        store_steps=20,
        energies=np.ndarray,
        save_animation=False,
        title="1D potential barrier",
        path_save=""):

    total_frames = int(fps * animation_duration)

    fig = plt.figure(figsize=figsize, facecolor='#002b36')
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("[Å]")
    ax.set_title("$\psi(x,t)$"+" "+title, color='white')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    time_ax = ax.text(0.97, 0.97, "",  color="white",
                      transform=ax.transAxes, ha="right", va="top")

    energy_ax = ax.text(0.97, 0.07, "", color='white',
                        transform=ax.transAxes, ha="right", va="top")

    plt.xlim(xlim)
    
    if ylim is not None:  # Correctly handle ylim
           plt.ylim(ylim)

    index = 0

    if Vmax-Vmin != 0:
        potential_plot = ax.plot(X/Å, (V + Vmin)/(Vmax-Vmin), label='$V(x)$')
    else:
        potential_plot = ax.plot(X/Å, V, label='$V(x)$')  
    real_plot, = ax.plot(X/Å, np.real(Ψ[index]), label='$Re|\psi(x)|$')
    imag_plot, = ax.plot(X/Å, np.imag(Ψ[index]), label='$Im|\psi(x)|$')
    abs_plot, = ax.plot(X/Å, np.abs(Ψ[index]), label='$|\psi(x)|$')

    ax.set_facecolor('#002b36')

    leg = ax.legend(facecolor='#002b36', loc='lower left')
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())

    xdt = np.linspace(0, total_time/femtoseconds, total_frames)
    psi_index = np.linspace(0, store_steps, total_frames)

    def func_animation(frame):

        index = int(psi_index[frame])

        energy_ax.set_text(u"energy = {} joules".format(
            "%.5e" % np.real(energies[index])))

        time_ax.set_text(u"t = {} femtoseconds".format(
            "%.3f" % (xdt[frame])))

        real_plot.set_ydata(np.real(Ψ[index]))
        imag_plot.set_ydata(np.imag(Ψ[index]))
        abs_plot.set_ydata(np.abs(Ψ[index]))

        return

    ani = animation.FuncAnimation(fig, func_animation,
                                  blit=False, frames=total_frames, interval=1/fps * 1000, cache_frame_data=False)
    if save_animation == True:
        if (title == ''):
            title = "animation"
        ani.save(path_save + title + '.gif',
                 fps=fps, metadata=dict(artist='Me'))
        plt.close(fig)
    else:
        plt.show()


def animate_2D(
        Ψ_plot,
        energies=np.ndarray,
        extent=10,
        V=np.ndarray,
        Vmin=-10,
        Vmax=10,
        xlim=np.ndarray,
        ylim=np.ndarray,
        figsize=(7, 7),
        animation_duration=5,
        fps=20,
        save_animation=False,
        potential_saturation=0.8,
        title="animation",
        path_save="",
        total_time=10,
        store_steps=20,
        wavefunction_saturation=1,
        rgba_or_rgb = True):

    total_frames = int(fps * animation_duration)

    figsize = (640*px, 640*px)

    fig = plt.figure(figsize=figsize, facecolor='#002b36')

    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    L = extent / const["Å"] / 2
    potential_plot = ax.imshow((V + Vmin)/(Vmax-Vmin),
                               vmax=1.0/potential_saturation, vmin=0, cmap=newcmp, origin="lower",
                               interpolation="gaussian", extent=[-L/2, L/2, -L/2, L/2])

    k = np.amax(np.abs(Ψ_plot[0])) * wavefunction_saturation
    if( rgba_or_rgb==True):
        wavefunction_plot = ax.imshow(complex_to_rgba(Ψ_plot[0], max_val=k),
                                  origin="lower", interpolation="gaussian", extent=[-L/2, L/2, -L/2, L/2])
    else:
        wavefunction_plot = ax.imshow(complex_to_rgb(Ψ_plot[0]),
                                  origin="lower", interpolation="gaussian", extent=[-L/2, L/2, -L/2, L/2])
    if xlim != None:
        ax.set_xlim(np.array(xlim)/const["Å"])
    if ylim != None:
        ax.set_ylim(np.array(ylim)/const["Å"])

    ax.set_title("$\psi(x,y,t)$"+" "+title, color="white")
    ax.set_xlabel('[Å]')
    ax.set_ylabel('[Å]')

    time_ax = ax.text(0.97, 0.97, "",  color="white",
                      transform=ax.transAxes, ha="right", va="top", alpha=0.9)

    energy_ax = ax.text(0.97, 0.93, "",  color="white",
                        transform=ax.transAxes, ha="right", va="top", alpha=0.9)

    xdt = np.linspace(0, total_time/const["femtoseconds"], total_frames)
    psi_index = np.linspace(0, store_steps, total_frames)

    def func_animation(frame):

        time_ax.set_text(
            u"time = {} femtoseconds".format("%.3f" % (xdt[frame])))

        index = int(psi_index[frame])
        k = np.amax(np.abs(Ψ_plot[index])) * wavefunction_saturation
        
        if( rgba_or_rgb==True):
            wavefunction_plot.set_data(complex_to_rgba(Ψ_plot[index], max_val=k))
        else:
             wavefunction_plot.set_data(complex_to_rgb(Ψ_plot[index]))
                
        formatted_num = "{:14.14e}".format(np.abs(energies[index]))
        energy_ax.set_text(u"Energy =  " + formatted_num)

        return wavefunction_plot, time_ax

    ani = animation.FuncAnimation(fig, func_animation,
                                  blit=True, frames=total_frames, interval=1/fps * 1000)
    if save_animation == True:
        title = title.replace("\n", " ")
        ani.save(path_save + title + '.gif', fps=fps, metadata=dict(artist='Me'))
        plt.close(fig)
    else:
        plt.show()


def plot_eigenstate_2D(eigenstates, energies, Ψ_plot, xlim=None, ylim=None, extent=10, wavefunction_saturation=0.5):

    eigenstates_array = eigenstates

    fig = plt.figure(figsize=(16/9 * 5.804 * 0.9, 5.804), facecolor='#002b36')

    grid = plt.GridSpec(2, 2, width_ratios=[4.5, 1], height_ratios=[
                        1, 1], hspace=0.1, wspace=0.2)
    ax1 = fig.add_subplot(grid[0:2, 0:1], facecolor='#002b36')
    ax2 = fig.add_subplot(grid[0:2, 1:2], facecolor='#002b36')

    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white')
    ax1.spines['left'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)

    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    ax2.spines['left'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)

    ax1.set_xlabel("$x$ [Å]")
    ax1.set_ylabel("$y$ [Å]")
    ax1.set_title("$\Psi(x,y)$", color='white')

    ax2.set_title('Energy Level', color='white')

    ax2.set_ylabel('$E_N$ [eV]')
    ax2.set_xticks(ticks=[])

    if xlim != None:
        ax1.set_xlim(np.array(xlim)/Å)
    if ylim != None:
        ax1.set_ylim(np.array(ylim)/Å)

    E0 = energies[0]

    for E in energies:
        ax2.plot([0, 1], [E, E], color='gray', alpha=0.5)

    ax2.plot([0, 1], [energies[k], energies[k]], color='yellow', lw=3)

    ax1.set_aspect('equal')
    L = extent/2/Å
    k = np.amax(np.abs(Ψ_plot))
    im = ax1.imshow(complex_to_rgba(eigenstates_array[Ψ_plot]*np.exp(1j*2*np.pi/10*k), max_val=k),
                    origin='lower', extent=[-L, L, -L, L],
                    interpolation='gaussian')
    plt.show()


def animate_superpositions_2D(energies: np.ndarray, eigenstates: np.ndarray,
                              extent=10, seconds_per_eigenstate=0.5,
                              fps=20, max_states=None, xlim=None, ylim=None, save_animation=False, title="animate_superpositions_2D",
                              path_save="", wavefunction_saturation=0.5):

    eigenstates /= np.max(np.abs(eigenstates))

    if max_states == None:
        max_states = len(energies)

    frames_per_eigenstate = fps * seconds_per_eigenstate
    total_time = max_states * seconds_per_eigenstate
    total_frames = int(fps * total_time)

    eigenstates_array = eigenstates

    figsize = (16/9 * 5.804 * 0.9, 5.804)
    fig = plt.figure(figsize=figsize, facecolor='#002b36')

    grid = plt.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[
                        1, 1], hspace=0.1, wspace=0.2)
    ax1 = fig.add_subplot(grid[0:2, 0:1], facecolor='#002b36')
    ax2 = fig.add_subplot(grid[0:2, 1:2], facecolor='#002b36')

    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white')
    ax1.spines['left'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)

    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    ax2.spines['left'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)

    ax1.set_xlabel("$x$ [Å]")
    ax1.set_ylabel("$y$ [Å]")

    ax1.set_title("$\Psi(x,y)$ "+title, color='white')

    ax2.set_title('Energy Level', color='white')

    ax2.set_ylabel('$E_N$ [eV]')
    ax2.set_xticks(ticks=[])

    if xlim != None:
        ax1.set_xlim(np.array(xlim)/Å)
    if ylim != None:
        ax1.set_ylim(np.array(ylim)/Å)

    E0 = energies[0]
    for E in energies:
        ax2.plot([0, 1], [E, E], color='gray', alpha=0.5)

    ax1.set_aspect('equal')
    L = extent/2/Å

    k = np.amax(np.abs(eigenstates_array[0]))
    eigenstate_plot = ax1.imshow(complex_to_rgba(eigenstates_array[0], max_val=k),
                                 origin='lower', extent=[-L, L, -L, L], interpolation='gaussian')

    line, = ax2.plot([0, 1], [energies[0], energies[0]], color='yellow', lw=1)

    plt.subplots_adjust(bottom=0.2)

    animation_data = {'n': 0.0}
    Δn = 1/frames_per_eigenstate

    def func_animation(*arg):
        animation_data['n'] = (animation_data['n'] + Δn) % len(energies)
        state = int(animation_data['n'])
        if (animation_data['n'] % 1.0) > 0.5:
            transition_time = (
                animation_data['n'] - int(animation_data['n']) - 0.5)
            eigenstate_combination = (np.cos(np.pi*transition_time)*eigenstates_array[state] +
                                      np.sin(np.pi*transition_time) *
                                      eigenstates_array[(state + 1) % len(energies)])

            k = np.amax(np.abs(eigenstate_combination))
            eigenstate_plot.set_data(complex_to_rgba(
                eigenstate_combination, max_val=k))

            E_N = energies[state]
            E_M = energies[(state + 1) % len(energies)]
            E = E_N*np.cos(np.pi*transition_time)**2 + E_M * \
                np.sin(np.pi*transition_time)**2
            line.set_ydata([E, E])
        else:
            line.set_ydata([energies[state], energies[state]])
            eigenstate_combination = eigenstates_array[int(state)]
            k = np.amax(np.abs(eigenstate_combination))
            eigenstate_plot.set_data(complex_to_rgba(
                eigenstate_combination, max_val=k))
        return eigenstate_plot, line

    ani = animation.FuncAnimation(fig, func_animation,
                                  blit=True, frames=total_frames, interval=1/fps * 1000)
    if save_animation == True:
        title = title.replace("\n", " ")
        ani.save(path_save + title + '.gif', fps=fps, metadata=dict(artist='Me'))
        plt.close(fig)
    else:
        plt.show()


def slider_plot_2D(eigenstates, energies, psi_0,  V,  p2,  hbar, m,  extent=10, xlim=None, ylim=None, dx = 1,  wavefunction_saturation=0.5):
   

    fig = plt.figure(figsize=(16/9 * 6.804 * 0.9, 6.804), facecolor='#002b36')

    grid = plt.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[
                        1, 1], hspace=0.1, wspace=0.2)
    ax1 = fig.add_subplot(grid[0:2, 0:1], facecolor='#002b36')
    ax2 = fig.add_subplot(grid[0:2, 1:2], facecolor='#002b36')
    
    # Get the position of the axes
    bbox = ax1.get_position()
    # Get the width of the axes
    width = bbox.x1 - bbox.x0
    # Get the height of the axes
    height = bbox.y1 - bbox.y0
    
    ax1.set_position([0.25, 0.18, width, height])
    # Get the position of the axes
    bbox = ax2.get_position()
    # Get the width of the axes
    width = bbox.x1 - bbox.x0
    # Get the height of the axes
    height = bbox.y1 - bbox.y0
    
    ax2.set_position([0.85, 0.18, width, height])
    # Note: The set_position() function takes a list of 4 parameters in the following order: [left, bottom, width, height]

    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white')
    ax1.spines['left'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)

    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    ax2.spines['left'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)

    ax1.set_xlabel("$x$ [Å]", color='white')
    ax1.set_ylabel("$y$ [Å]", color='white')
    ax1.set_title("$\Psi(x,y)$", color='white')

    ax2.set_title('Energy Level', color='white')

    ax2.set_ylabel('$E_N$ [eV]', color='white')
    ax2.set_xticks(ticks=[])

    coeffs = np.abs(np.tensordot(eigenstates.conj(), psi_0))
    indices = np.array([i for i in range(coeffs.shape[0]) if coeffs[i] < 0.1])
    str_coeffs = np.array2string(indices, separator=",")
    
    text_kwargs = dict(fontsize=14, color='white')
    fig.text(0.17, 0.05, str_coeffs, **text_kwargs)

    if xlim != None:
        ax1.set_xlim(np.array(xlim)/Å)
    if ylim != None:
        ax1.set_ylim(np.array(ylim)/Å)

    E0 = energies[0]
    for E in energies:
        ax2.plot([0, 1], [E, E], color='gray', alpha=0.5)

    ax1.set_aspect('equal')
    L = extent/2/Å
    k = np.amax(np.abs(eigenstates[0]))
    eigenstate_plot = ax1.imshow(complex_to_rgba(eigenstates[0], max_val=k),
                                 origin='lower', extent=[-L, L, -L, L],
                                 interpolation='gaussian')


   # plt.subplots_adjust(bottom=0.2)
    
    enable_slider1 = True
    enable_slider2 = True
    enable_slider3 = False

    line1 = ax2.plot([0, 1], [energies[0], energies[0]], color='yellow', lw=1)
    line2 = ax2.plot([0, 1], [energies[0], energies[0]], color='orange', lw=1)
    line3 = ax2.plot([0, 1], [energies[0], energies[0]], color='red', lw=1)
    
    energy = np.abs(Energies(V, [eigenstates[5]], p2, hbar, m))
    line4 = ax2.plot([0, 1], [energy, energy], color='magenta', lw=1)
    

    button1_ax = plt.axes([0.005, 0.95, 0.05, 0.04]) # Define the position and size of the button
    button1 = Button(button1_ax, "Enable", color = 'green') # Create the button object
    button1.hovercolor = 'green'
    
    button2_ax = plt.axes([0.06, 0.95, 0.05, 0.04]) # Define the position and size of the button
    button2 = Button(button2_ax, "Enable", color = 'green') # Create the button object
    button2.hovercolor = 'green'
    
    button3_ax = plt.axes([0.115, 0.95, 0.05, 0.04]) # Define the position and size of the button
    button3 = Button(button3_ax, "Enable", color = 'red') # Create the button object
    button3.hovercolor = 'red'
    
    slider1_ax = plt.axes([0.005, 0.05, 0.05, 0.85], facecolor="#0038b8")
    slider2_ax = plt.axes([0.06, 0.05, 0.05, 0.85], facecolor="#0038b8")
    slider3_ax = plt.axes([0.115, 0.05, 0.05, 0.85], facecolor="#0038b8")
    #[left, bottom, width, height]


    slider1 = Slider(slider1_ax,      # the axes object containing the slider
                    'state',            # the name of the slider parameter
                    0,          # minimal value of the parameter
                    # maximal value of the parameter
                    len(eigenstates)-1,
                    valinit= 31,  # initial value of the parameter  #31
                    valstep=1,
                    color='gray',
                    orientation='vertical'
                    )
                    
    slider1_val_line = slider1_ax.axhline(y=slider1.val, color='yellow', linewidth=2)
    
    slider2 = Slider(slider2_ax,      # the axes object containing the slider
                    'state',            # the name of the slider parameter
                    0,          # minimal value of the parameter
                    # maximal value of the parameter
                    len(eigenstates)-1,
                    valinit= 13,  # initial value of the parameter   #15
                    valstep=1,
                    color='gray',
                    orientation='vertical'
                    )

    slider2_val_line = slider2_ax.axhline(y=slider2.val, color='orange', linewidth=2)
    
    slider3 = Slider(slider3_ax,      # the axes object containing the slider
                    'state',            # the name of the slider parameter
                    0,          # minimal value of the parameter
                    # maximal value of the parameter
                    len(eigenstates)-1,
                    valinit= 13,  # initial value of the parameter
                    valstep=1,
                    color='gray',
                    orientation='vertical'
                    )

    slider3_val_line = slider3_ax.axhline(y=slider3.val, color='red', linewidth=2)
    
    
    def update1(state):
        state = int(state)
        if(enable_slider1 == True):
            s1 = "|ψ" + str(state) + "⟩"
            s2 = "|ψ" + str(slider2.val) + "⟩"
            s3 = "|ψ" + str(slider3.val) + "⟩"
            title = s1
            slider1_val_line.set_ydata(state)
            T = eigenstates[state]
            if(enable_slider2 == True):
                T = np.tensordot(eigenstates[slider1.val], eigenstates[slider2.val], axes = 1)
            if(enable_slider3 == True):
                T = np.tensordot(T, eigenstates[slider3.val], axes = 1)
      
            if(enable_slider2 == True and enable_slider3 == True ):
                title = title + "+" + s2 + "+" + s3
            elif(enable_slider2 == True):
                 title = title  + "+" + s2
            elif(enable_slider3 == True):
                title = title + "+" + s3
            
            k = np.amax(np.abs(T))
            if (k>1e-3):
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=k))
            else:
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=1))
                
            line1[0].set_ydata([energies[state], energies[state]])
            
            energy = np.abs(Energies(V, [T], p2, hbar, m))  / 2
            line4[0].set_ydata([energy, energy])
           
            
            ax1.set_title("$\Psi(x,y)$ " + title, color='white')
        
        
    def update2(state):
        state = int(state)
        if(enable_slider2 == True):
            s1 = "|ψ" + str(slider1.val) + "⟩"
            s2 = "|ψ" + str(state) + "⟩"
            s3 = "|ψ" + str(slider3.val) + "⟩"
            title = s2
            T = eigenstates[state]
            slider2_val_line.set_ydata(state)
            if(enable_slider1 == True):
                T = np.tensordot(eigenstates[slider1.val], eigenstates[slider2.val], axes = 1)
            if(enable_slider3 == True):
                T = np.tensordot(T, eigenstates[slider3.val], axes = 1)
           
            if(enable_slider1 == True and enable_slider3 == True ):
                title = s1 + "+" + title + "+" + s3
            elif(enable_slider1 == True):
                 title = s1  + "+" + title
            elif(enable_slider3 == True):
                title = title + "+" + s3
            
            k = np.amax(np.abs(T))
            if (k>1e-3):
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=k))
            else:
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=1))
                
            line2[0].set_ydata([energies[state], energies[state]])
            
            energy = np.abs(Energies(V, [T], p2, hbar, m)) / 2
            line4[0].set_ydata([energy, energy])
            
            ax1.set_title("$\Psi(x,y)$ " + title, color='white')


    def update3(state):
        state = int(state)
        if(enable_slider3 == True):
            s1 = "|ψ" + str(slider1.val) + "⟩"
            s2 = "|ψ" + str(slider2.val) + "⟩"
            s3 = "|ψ" + str(state) + "⟩"
            title = s3
            T = eigenstates[state]
            slider3_val_line.set_ydata(state)
            if(enable_slider1 == True  and  enable_slider2 == True):
                T = np.tensordot(eigenstates[slider1.val], eigenstates[slider2.val], axes = 1)
                T = np.tensordot(T, eigenstates[slider3.val], axes = 1)
                title = s1 + "+" + s2 + "+" + title
            elif (enable_slider1 == True):
                title = s1 + "+" + title
            elif (enable_slider2 == True):
                title = s2 + "+" + title
            
            k = np.amax(np.abs(T))
            if (k>1e-3):
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=k))
            else:
                eigenstate_plot.set_data(complex_to_rgba(T, max_val=1))
            
            line3[0].set_ydata([energies[state], energies[state]])
            
            energy = np.abs(Energies(V, [T], p2, hbar, m))  / 2
            line4[0].set_ydata([energy, energy])
            
            ax1.set_title("$\Psi(x,y)$ " + title, color='white')

    def on_button1_clicked(event):
        nonlocal enable_slider1
        enable_slider1 = not  enable_slider1
        if(enable_slider1 == True):
                button1.color = "green"
        else:
                button1.color = "red"
            

    def on_button2_clicked(event):
        nonlocal enable_slider2
        enable_slider2 = not  enable_slider2
        if(enable_slider2 == True):
                button2.color = "green"
        else:
                button2.color = "red"
             

    def on_button3_clicked(event):
        nonlocal enable_slider3
        enable_slider3 = not  enable_slider3
        if(enable_slider3 == True):
                button3.color = "green"
        else:
                button3.color = "red"
             

    button1.on_clicked(on_button1_clicked) # Connect the button to the callback function
    button2.on_clicked(on_button2_clicked) # Connect the button to the callback function
    button3.on_clicked(on_button3_clicked) # Connect the button to the callback function
    
    slider1.label.set_color('white')
    slider1.valtext.set_color('white')
    slider2.label.set_color('white')
    slider2.valtext.set_color('white')
    slider3.label.set_color('white')
    slider3.valtext.set_color('white')
    
    slider1.hline.set_alpha(0)
    slider2.hline.set_alpha(0)
    slider3.hline.set_alpha(0)
    
    update1(slider1.val)
    slider1.on_changed(update1)
    
    update2(slider2.val)
    slider2.on_changed(update2)
    
    update3(slider3.val)
    slider3.on_changed(update3)
    
    plt.show()


def slider_plot_2D_1(eigenstates, energies, extent=10, xlim=None, ylim=None, wavefunction_saturation=0.5):

    eigenstates_array = normalize(eigenstates)

    fig = plt.figure(figsize=(16/9 * 5.804 * 0.9, 5.804), facecolor='#002b36')

    grid = plt.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[
                        1, 1], hspace=0.1, wspace=0.2)
    ax1 = fig.add_subplot(grid[0:2, 0:1], facecolor='#002b36')
    ax2 = fig.add_subplot(grid[0:2, 1:2], facecolor='#002b36')

    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white')
    ax1.spines['left'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)

    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    ax2.spines['left'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)

    ax1.set_xlabel("$x$ [Å]", color='white')
    ax1.set_ylabel("$y$ [Å]", color='white')
    ax1.set_title("$\Psi(x,y)$", color='white')

    ax2.set_title('Energy Level', color='white')

    ax2.set_ylabel('$E_N$ [eV]', color='white')
    ax2.set_xticks(ticks=[])

    if xlim != None:
        ax1.set_xlim(np.array(xlim)/Å)
    if ylim != None:
        ax1.set_ylim(np.array(ylim)/Å)

    E0 = energies[0]
    for E in energies:
        ax2.plot([0, 1], [E, E], color='gray', alpha=0.5)

    ax1.set_aspect('equal')
    L = extent/2/Å

    k = np.amax(np.abs(eigenstates_array[0]))
    eigenstate_plot = ax1.imshow(complex_to_rgba(eigenstates_array[0], max_val=k),
                                 origin='lower', extent=[-L, L, -L, L],
                                 interpolation='gaussian')

    line = ax2.plot([0, 1], [energies[0], energies[0]], color='yellow', lw=1)

    plt.subplots_adjust(bottom=0.2)

    slider_ax = plt.axes([0.2, 0.05, 0.7, 0.05], facecolor="#0038b8")
    

    slider = Slider(slider_ax,      # the axes object containing the slider
                    'state',            # the name of the slider parameter
                    0,          # minimal value of the parameter
                    # maximal value of the parameter
                    len(eigenstates_array)-1,
                    valinit=1,  # initial value of the parameter
                    valstep=1,
                    color='gray'
                    )

    
    slider_val_line = slider_ax.axvline(x=slider.val, color='red', linewidth=2)
    
    def update(state):
        state = int(state)
        slider_val_line.set_data([state, slider_val_line.get_ydata()])
        k = np.amax(np.abs(eigenstates_array[state]))
        eigenstate_plot.set_data(complex_to_rgba(
            eigenstates_array[state], max_val=k))
        line[0].set_ydata([energies[state], energies[state]]) 
    
    
    slider.label.set_color('white')
    slider.valtext.set_color('white')
    
    slider.vline.set_alpha(0)
    
    update(slider.val)

    slider.on_changed(update)
    plt.show()

def superpositions_2D(extent, states, eigenstates, energies, fps=30, total_time=20,
                      dt=0.001, xlim=[-10/2, 10/2], ylim=[-10/2, 10/2], save_animation=False, hide_controls=False, title="animate_superpositions_2D",
                      path_save=""):

    total_frames = fps * total_time

    eigenstates /= np.max(np.abs(eigenstates))

    coeffs = None
    if isinstance(states, int) or isinstance(states, float):
        coeffs = np.array([1.0 if i == 0 else 0.0 for i in range(states)],
                          dtype=np.complex128)
        eigenstates = eigenstates[0: states]
    else:
        coeffs = states
        eigenstates = eigenstates[0: len(states)]
        states = len(states)

    N = eigenstates.shape[1]

    figsize = (640*px, 640*px)
    
    #fig = plt.figure(figsize=(16/9 * 7.804 * 0.9, 7.804), facecolor='#002b36')
    fig = plt.figure(figsize=figsize, facecolor='#002b36')
    
    
    grid_width = 10
    grid_length = states if states < 30 else 30
    grid = plt.GridSpec(grid_width, grid_length)
    grid_slice = grid[0:int(0.7*grid_width), 0:grid_length]
    if hide_controls:
        grid_slice = grid[0:grid_width, 0:grid_length]
    ax = fig.add_subplot(grid_slice, facecolor='#002b36')

    ax.set_title("$\psi(x, y)$"+" "+title, color='white')
    ax.set_xlabel("$x$ [Å]", color='white')
    ax.set_ylabel("$y$ [Å]", color='white')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    # ax.set_xticks([])
    # ax.set_yticks([])
    def get_norm_factor(psi): return 1.0/np.sqrt(np.sum(psi*np.conj(psi))+1e-6)
    coeffs = np.array(coeffs, dtype=np.complex128)
    X, Y = np.meshgrid(np.linspace(-1.0, 1.0, eigenstates[0].shape[0]),
                       np.linspace(-1.0, 1.0, eigenstates[0].shape[1]))
    maxval = np.amax(np.abs(eigenstates[0]))

    ax.set_xlim(np.array(xlim)/Å)
    ax.set_ylim(np.array(ylim)/Å)
    
    k = np.amax(np.abs(eigenstates[0]))
    
    im = plt.imshow(complex_to_rgba(eigenstates[0], max_val=k), interpolation='gaussian',
                    origin='lower', extent=[-extent/2/Å,
                                            extent/2/Å,
                                            -extent/2/Å,
                                            extent/2/Å]
                    )
    # im2 = plt.imshow(0.0*eigenstates[0], cmap='gray')
    animation_data = {'ticks': 0, 'norm': 1.0}

    def make_update(n):
        def update(phi, r):
            coeffs[n] = r*np.exp(1.0j*phi)
            psi = np.dot(coeffs,
                         eigenstates.reshape([states, N*N]))
            psi = psi.reshape([N, N])
            animation_data['norm'] = get_norm_factor(psi)
            psi *= animation_data['norm']
            # apsi = np.abs(psi)
            # im.set_alpha(apsi/np.amax(apsi))
        return update

    widgets = []
    circle_artists = []
    if not hide_controls:
        for i in range(states):
            if states <= 30:
                circle_ax = fig.add_subplot(
                    grid[8:10, i], projection='polar', facecolor='#002b36')
                circle_ax.set_title(str(i)  # + '\nE=' + str() + '$E_0$'
                                    , size=8.0 if states < 15 else 6.0, color='white'
                                    )
            else:
                circle_ax = fig.add_subplot(grid[8 if i < 30 else 9,
                                                 i if i < 30 else i-30],
                                            projection='polar')
                circle_ax.set_title(str(i)  # + '\nE=' + str() + '$E_0$'
                                    , size=8.0 if states < 15 else 6.0, color='white'
                                    )
            circle_ax.set_xticks([])
            circle_ax.set_yticks([])
            widgets.append(ComplexSliderWidget(
                circle_ax, 0.0, 1.0, animated=True))
            widgets[i].on_changed(make_update(i))
            circle_artists.append(widgets[i].get_artist())
    artists = circle_artists + [im]

    def func(*args):
        animation_data['ticks'] += 1
        e = np.exp(-1.0j*energies[0:states]*dt)
        np.copyto(coeffs, coeffs*e)
        norm_factor = animation_data['norm']
        psi = np.dot(coeffs*norm_factor,
                     eigenstates.reshape([
                         states, N*N]))
        psi = psi.reshape([N, N])
        
        k = np.amax(np.abs(psi))
        im.set_data(complex_to_rgba(psi, max_val=k))
        # apsi = np.abs(psi)
        # im.set_alpha(apsi/np.amax(apsi))
        # if animation_data['ticks'] % 2:
        #     return (im, )
        # else:
        if not hide_controls:
            for i, c in enumerate(coeffs):
                phi, r = np.angle(c), np.abs(c)
                artists[i].set_xdata([phi, phi])
                artists[i].set_ydata([0.0, r])
        return artists

    ani = animation.FuncAnimation(fig, func, blit=True, interval=1/fps * 1000,
                                  frames=None if (not save_animation) else
                                  total_frames)
    if save_animation == True:
        if (title == ''):
            title = "animation"
        ani.save(path_save + title + '.gif',
                 fps=fps, metadata=dict(artist='Me'))
        plt.close(fig)
    else:
        plt.show()
        plt.show()
