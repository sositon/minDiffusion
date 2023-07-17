import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap, to_rgba

# Globals and hyper parameters
PATH = "/Users/omersiton/PycharmProjects/DiffusionModel"
epochs = 3000
lr = 1e-5


#########################################################
####################    Models    #######################
#########################################################
class UnconditionalDenoiser(nn.Module):
    def __init__(self, latent_dim=256):
        super(UnconditionalDenoiser, self).__init__()
        self.fc1 = nn.Linear(3, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        # Predicts added noise and not X0
        self.fc4 = nn.Linear(latent_dim, 2)

    def forward(self, x_t, t):
        t = torch.ones_like(x_t[:, :1]) * t
        x_t = torch.cat([x_t, t], dim=-1)
        x_t = nn.LeakyReLU()(self.fc1(x_t))
        x_t = nn.LeakyReLU()(self.fc2(x_t))
        return self.fc4(x_t)


class ConditionalDenoiser(nn.Module):
    def __init__(self, latent_dim=256, num_classes=5, embedding_dim=5):
        super(ConditionalDenoiser, self).__init__()
        # Embedding layer to transform class labels into higher-dimensional space
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.fc1 = nn.Linear(3 + embedding_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 2)

    def forward(self, x_t, t, class_label):
        # Extend dimensions of 't' and 'class_label' to match 'x_t'
        t = torch.ones_like(x_t[:, :1]) * t
        class_label = torch.ones_like(x_t[:, :1]) * class_label

        # Convert class labels into higher-dimensional embeddings
        class_embedding = self.embedding(class_label.type(torch.int)).squeeze(1)

        x_t = torch.cat([x_t, t, class_embedding], dim=-1).type(torch.float32)
        x_t = nn.LeakyReLU()(self.fc1(x_t))
        x_t = nn.LeakyReLU()(self.fc2(x_t))
        return self.fc3(x_t)


class Points2D(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'data': torch.tensor(self.data[index]),
            'target': torch.tensor(self.targets[index])
        }
        return sample['data'], sample['target']


#########################################################
#############    Helper Functions - Shared  #############
#########################################################
def sigma(t):
    return np.exp(5 * (t - 1))


def sigma_derivative(t):
    return 5 * sigma(t)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def SNR(t):
    return 1 / sigma(t) ** 2


#########################################################
#######    Helper Functions - Unconditional Part  #######
#########################################################


def plot_samples(points):
    # Used GPT
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("2D points uniformly distributed in a square")
    plt.show()


def plot_trajectories(trajectories, indices=[0,1,2,3]):
    # Used GPT
    for i in indices:
        # Extract x and y coordinates
        x = trajectories[i, :, :, 0]
        y = trajectories[i, :, :, 1]

        # Plot the trajectory
        plt.scatter(x, y, label=f'Trajectory {i + 1}', s=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Denoising Trajectories')
    plt.legend()
    plt.show()


def plot_trajectory(trajectory):
    # Used GPT
    T = 1000
    plt.figure(figsize=(8, 8))
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.linspace(0, 1, T), cmap=get_cmap('jet'), s=3)
    plt.colorbar(label='Time (t)')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.title('Forward process as a trajectory in 2D space')
    plt.show()


def forward_process(x0):
    trajectory = [x0]
    T = 1000
    dt = 1.0 / T
    times = np.arange(0, 1, dt)
    for t in times[1:]:
        eps = np.random.normal(size=x0.shape)
        x_next = trajectory[-1] + sigma(t) * eps
        trajectory.append(x_next)

    return np.array(trajectory)


def sample_reverse_process_DDIM(denoiser, T=1000, xt=None, c=None):
    dt = -1 / T
    xt = torch.randn(size=[1, 2]) if xt is None else xt.clone()
    trajectory = []

    for t in np.linspace(1, 0, T):
        trajectory.append(xt.clone().detach().numpy())
        noise_prediction = denoiser(xt, t) if c is None else denoiser(xt, t, c)
        x0 = xt - (sigma(t) * noise_prediction)  # Eq. 8
        score_xt = (x0 - xt) / (sigma(t) ** 2)
        dx = -sigma_derivative(t) * sigma(t) * score_xt * dt  # Eq. 7
        xt += dx

    return xt.clone(), trajectory


def sample_reverse_process_DDPM(denoiser, T=1000, xt=None, lambda_=0.1):
    dt = -1 / T
    xt = torch.randn(size=[1, 2]) if xt is None else xt.clone()
    trajectory = []

    for t in np.linspace(1, 0, T):
        trajectory.append(xt.clone().detach().numpy())
        noise_prediction = denoiser(xt, t)
        x0 = xt - (sigma(t) * noise_prediction)  # Eq. 8
        score_xt = (x0 - xt) / (sigma(t) ** 2)
        # Eq. 7 - DDPM version
        dx = -(1 + lambda_**2) * sigma_derivative(t) * sigma(t) * score_xt * dt
        # Add the stochastic term
        gt = np.sqrt(2 * sigma_derivative(t) * sigma(t))
        dx += lambda_ * gt * torch.normal(0, std=lambda_, size=xt.shape)

        xt += dx

    return xt.clone(), trajectory


def estimate_log_probability(x, denoiser, T=1000, num_combinations=1000, c=None):
    SNR_diffs = []
    for _ in range(num_combinations):
        noise = torch.randn_like(x)
        t = torch.rand(1)
        xt = x + sigma(t) * noise
        noise_prediction = denoiser(xt, t) if c is None else denoiser(xt, t, c)
        x0 = xt - sigma(t) * noise_prediction
        snr_t = SNR(t)
        snr_t_dt = SNR(t - 1 / T)
        snr_diff = snr_t_dt - snr_t
        sq_distance = nn.MSELoss()(x, x0).item()
        SNR_diffs.append(snr_diff * sq_distance)

    log_prob_estimate = (T/2) * np.stack(SNR_diffs, axis=0).mean()

    return -log_prob_estimate


def train_uncond(denoiser, data, optimizer, criterion):
    denoiser.train()
    train_loss = []
    for epoch in range(epochs):
        # we use all data as our batch
        optimizer.zero_grad()
        eps = torch.randn_like(data)
        t = torch.rand([data.size(0), 1])
        x_t = data + sigma(t) * eps
        noise_pred = denoiser(x_t, t)
        loss = criterion(noise_pred, eps)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss {loss.item()}')

    return train_loss


def plot_DDIM_sampling_with_different_seeds(model, num_samples):
    # Used GPT
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))  # Create a 3x3 grid of subplots
    for i in range(3):
        for j in range(3):
            set_seed(i * 3 + j)
            noise = torch.randn(size=[num_samples, 1, 2])
            axs[i, j].scatter(noise[:, :, 0], noise[:, :, 1], s=1)
            axs[i, j].set_xlim([-3, 3])
            axs[i, j].set_ylim([-3, 3])
            axs[i, j].set_title(f'Seed: {i * 3 + j}')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(3):
        for j in range(3):
            set_seed(i * 3 + j)
            samples = [sample_reverse_process_DDIM(model)[0].detach() for _ in range(num_samples)]
            samples = torch.stack(samples)

            # Plot the points
            axs[i, j].scatter(samples[:, :, 0], samples[:, :, 1], s=1)
            axs[i, j].set_xlim([-3, 3])
            axs[i, j].set_ylim([-3, 3])
            axs[i, j].set_title(f'Seed: {i * 3 + j}')
            print(f"{i * 3 + j}")
    plt.tight_layout()
    plt.show()


def plot_diffusion_samples_with_different_Ts(model, T_values, num_samples):
    # Used GPT
    fig, axs = plt.subplots(2, 3, figsize=(9, 9))
    for i in range(2):
        for j in range(3):
            T = T_values[i * 3 + j]
            samples = [sample_reverse_process_DDIM(model, T=T)[0].detach() for _ in range(num_samples)]
            samples = torch.stack(samples)
            # Plot the points
            axs[i, j].scatter(samples[:, :, 0], samples[:, :, 1], s=1)
            axs[i, j].set_xlim([-3, 3])
            axs[i, j].set_ylim([-3, 3])
            axs[i, j].set_title(f'T: {T}')
    plt.tight_layout()
    plt.show()


def modify_sampler(modified_sigma, dt=1e-2):
    # Used GPT
    t_values = np.arange(0, 1, dt)
    original_sigma_values = [sigma(t) for t in t_values]
    plt.plot(t_values, original_sigma_values, label='Original')
    modified_sigma_values = [modified_sigma(t) for t in t_values]
    plt.plot(t_values, modified_sigma_values, label='Modified')
    plt.xlabel('Time')
    plt.ylabel('Sigma')
    plt.legend()
    plt.show()


def q6(model, input_noise):
    # Used GPT
    # DDIM sampling
    outputs_DDIM = [sample_reverse_process_DDIM(model, T=1000, xt=input_noise)[0].detach().numpy()
                    for _ in range(10)]
    outputs_DDIM = np.stack(outputs_DDIM, axis=0)

    # Plot the outputs
    for i in range(10):
        plt.scatter(outputs_DDIM[i, :, 0], outputs_DDIM[i, :, 1], label=f'Output {i + 1}')
    plt.legend()
    plt.title("DDIM")
    plt.show()

    # DDPM sampling
    outputs_DDPM, trajectories = [], []
    for _ in range(10):
        s, traj = sample_reverse_process_DDPM(model, T=1000, xt=input_noise)
        outputs_DDPM.append(s.clone().detach().numpy())
        trajectories.append(traj)
    outputs_DDPM = np.stack(outputs_DDPM, axis=0)
    trajectories = np.stack(trajectories, axis=0)
    # Plot the outputs
    for i in range(10):
        plt.scatter(outputs_DDPM[i, :, 0], outputs_DDPM[i, :, 1], label=f'Output {i + 1}')
    plt.legend()
    plt.title("DDPM")
    plt.show()
    plot_trajectories(trajectories)


def guided_sample(denoiser, target_point, T=1000):
    dt = -1 / T
    xt = torch.rand(size=[1, 2])

    for t in np.linspace(1, 0, T):
        noise_prediction = denoiser(xt, t)
        x0 = xt - (sigma(t) * noise_prediction)  # Eq. 8
        score_xt = (x0 - xt) / (sigma(t)**2)
        dx = -sigma_derivative(t)*sigma(t)*score_xt*dt  # Eq. 7
        guide_vector = target_point - xt
        guide_vector = guide_vector / torch.norm(guide_vector)
        guided_noise = guide_vector * torch.norm(dx)

        xt += dx + guided_noise

    return xt


#########################################################
########    Helper Functions - Conditional Part  ########
#########################################################
def assign_class(point):
    """left it as a function and not a simple dict as a design choice.
    If for some reason in the future I will want to change the classes to more complex classes"""
    x, y = point
    if y < -0.6:
        return 0
    elif y < -0.2:
        return 1
    elif y < 0.2:
        return 2
    elif y < 0.6:
        return 3
    else:
        return 4


def points_colored_by_class(classes):
    x, y = points[:, 0], points[:, 1]
    # Create a scatter plot.
    color_background()
    plt.scatter(x, y, c=cmap(classes))
    # Show the plot.
    plt.show()


def train_cond(denoiser, dataloader, optimizer, criterion):
    denoiser.train()
    train_loss = []
    for epoch in range(epochs):
        for data, c in dataloader:
            # we use all data as our batch
            optimizer.zero_grad()
            eps = torch.randn_like(data)
            t = torch.rand([data.size(0), 1])
            x_t = data + sigma(t) * eps
            noise_pred = denoiser(x_t, t, c.view(data.size(0), 1))
            loss = criterion(noise_pred, eps.type(torch.float32))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if epoch % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss {loss.item()}')

    return train_loss


def q3_cond():
    # Used GPT
    trajectories_x = [[] for _ in range(num_classes)]
    trajectories_y = [[] for _ in range(num_classes)]
    # Sample one point from each class and get its trajectory.
    for i in range(num_classes):
        point, traj = sample_reverse_process_DDIM(model_cond, c=i)
        traj = np.array(traj)
        trajectories_x[i] = traj[:, :, 0]
        trajectories_y[i] = traj[:, :, 1]
    color_background()
    # Plot the trajectories on top
    for i in range(num_classes):
        # Scatter plot
        plt.scatter(trajectories_x[i], trajectories_y[i], color=cmap(i), label=f"class {i}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()


def color_background():
    # Used GPT
    # Generate a grid of points over the plot area
    x_range = np.linspace(-1, 1, 200)
    y_range = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(x_range, y_range)
    # Predict the class for each point on the grid
    Z = np.array([assign_class([x, y]) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    # Plot the predicted classes as the background color
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


def q4_cond():
    # Used GPT
    samples = []
    for i in range(num_classes):
        samples.append(
            [sample_reverse_process_DDIM(model_cond, c=i)[0].detach().numpy() for _ in range(40)])
    samples = np.stack(samples, axis=0)
    color_background()
    for i in range(num_classes):
        plt.scatter(samples[i, :, :, 0], samples[i, :, :, 1], color=cmap(i), label=f"class {i}")
    plt.legend()
    plt.show()


def q6_cond():
    # Used GPT
    points = [
        [[0.5, -0.8]],  # class 0
        [[-0.5, -0.8]],  # class 0
        [[-2.0, -1.2]],  # Outside any class
        [[-2.0, -1.1]],  # Outside any class
        [[0.2, 0.0]],  # Mismatching location and class
        [[0.25, 0.0]],  # Mismatching location and class
    ]
    classes = [
        [0],  # class 0
        [0],  # class 0
        [0],  # Outside any class closer class
        [4],  # Outside any class farther class
        [2],  # good class
        [3],  # bad class
    ]
    points = torch.tensor(points, dtype=torch.float32)
    # Estimate the log-probabilities of the points
    log_probs = [estimate_log_probability(x, model_cond, c=c[0]) for x, c in zip(points, classes)]
    color_background()
    # Plot the points on top of the data
    for i in range(len(points)):
        x, y = points[i][0]
        log_prob = log_probs[i]
        plt.plot(x, y, 'o', label=f'Log Prob: {log_prob:.2f}', color=cmap(classes[i][0]))
    plt.legend()
    plt.xlim(-2.1, 2)
    plt.ylim(-2, 2)
    plt.show()


#########################################################
####################     Main     #######################
#########################################################


TRAIN = False
if __name__ == '__main__':
    set_seed(0)
    num_points = 2000  # Or any number between 1000 and 3000
    points = np.random.uniform(-1, 1, (num_points, 2))
    plot_samples(points)

    # q2.2.2.1:
    # x0 = np.array([0.0, 0.0])
    # trajectory = forward_process(x0)
    # plot_trajectory(trajectory)

    # q2.2.2.2:
    model = UnconditionalDenoiser()
    points = torch.from_numpy(points).float()
    if TRAIN:
        optimizer = optim.Adam(model.parameters(), lr)
        criterion = nn.MSELoss()
        train_loss = train_uncond(model, points, optimizer, criterion)
        # plot the training loss
        plt.figure(figsize=(8, 8))
        plt.plot(train_loss)
        plt.title('Training loss convergence process')
        plt.show()
        # save the trained model
        # torch.save(model.state_dict(), f"{PATH}/model.pt")

    # # load the saved model
    # model.load_state_dict(torch.load(f"{PATH}/model.pt"))
    # model.eval()

    # q2.2.2.3:
    # plot_DDIM_sampling_with_different_seeds(num_samples=1000)
    # set seed back to how we started
    # set_seed(0)

    # q2.2.2.4:
    # T_values = [1, 10, 50, 100, 500, 1000]
    # plot_diffusion_samples_with_different_Ts(T_values, num_samples=1000)

    # q2.2.2.5:
    # modify_sampler(lambda t: np.exp(3*(t-1)))

    # q2.2.2.6:
    # input_noise = torch.randn(1, 2)  # This is the same input noise we'll use every time
    # q6(input_noise)

    # sample points
    sampled_points = [sample_reverse_process_DDIM(model) for _ in range(100)]

    # Conditional Part Globals
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    num_classes = 5
    light_colors = [(r, g, b, 0.2) for r, g, b, _ in [to_rgba(c) for c in colors]]
    cmap = ListedColormap(colors[:num_classes])
    cmap_light = ListedColormap(light_colors[:num_classes])

    # q2.2.2.1
    targets = [assign_class(point) for point in points]
    # points_colored_by_class(targets)

    # q2.2.2.2
    model_cond = ConditionalDenoiser()
    dataset = Points2D(points, targets)
    dataloader = DataLoader(dataset, batch_size=num_points)
    if TRAIN:
        optimizer = optim.Adam(model_cond.parameters(), lr)
        criterion = nn.MSELoss()
        train_loss = train_cond(model_cond, dataloader, optimizer, criterion)
        # plot the training loss
        plt.figure(figsize=(8, 8))
        plt.plot(train_loss)
        plt.title('Training loss convergence process')
        plt.show()
        # save the trained model_cond
        # torch.save(model_cond.state_dict(), f"{PATH}/model_cond.pt")

    # load the saved model_cond
    # model_cond.load_state_dict(torch.load(f"{PATH}/model_cond.pt"))
    # model_cond.eval()

    # q2.2.2.3
    q3_cond()

    # q2.2.2.4
    # q4_cond()

    # q2.2.2.6
    # Create some points
    # q6_cond()

    # Bonus - it's not even close
    # sample = guided_sample(model, target_point=torch.tensor([[4, 4.5]]))
