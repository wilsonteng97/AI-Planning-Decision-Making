import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 2000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
        len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.capacity = buffer_limit
        self.buffer = list()
    
    def push(self, transition):
        '''
        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        if len(self.buffer) >= self.capacity:
            # if buffer is full, remove the oldest the transition
            self.buffer.pop(0)
        # add new transition experience at the end of buffer
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        '''
        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        rewards = [t.reward[0] for t in self.buffer]

        # min-max normalisation (signed integer to 0-1 range)
        max_r = max(rewards)
        min_r = min(rewards)
        range_r = max_r - min_r
        normalised_reward = [(float(r) - min_r) / range_r for r in rewards]
        
        # weighted sampling
        res = random.choices(self.buffer, weights=normalised_reward, k=batch_size)
        
        res_state = [r.state for r in res]
        res_action = [r.action for r in res]
        res_reward = [r.reward for r in res]
        res_next_state = [r.next_state for r in res]
        res_done = [r.done for r in res]
        
        states = torch.tensor(res_state, dtype=torch.float, device=device)
        actions = torch.tensor(res_action, device=device)
        rewards = torch.tensor(res_reward, dtype=torch.float, device=device)
        next_states = torch.tensor(res_next_state, dtype=torch.float, device=device)
        dones = torch.tensor(res_done, dtype=torch.float, device=device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)


class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        prob = random.random()
        val = 0

        if prob < epsilon:
            # random exploration
            val = random.randrange(self.num_actions)
        else:
            # actual action
            actions = self.forward(state)
            # choose the action with the best expected return
            val = np.argmax(actions.cpu().data.numpy())
        
        return int(val)


class DQN(BaseAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )


class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()


class AtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.
    '''

    # Huber's loss
    criterion = torch.nn.SmoothL1Loss()

    # gather predictions for each Q(s,a) in the batch job
    predictions = model(states).gather(1, actions)

    # (reward + discount factor * maxQ(s',a))
    nexts = target(next_states).detach().max(1)[0].unsqueeze(1)
    # only reward term for terminal state (done = true/1)
    labels = rewards + gamma * nexts * (1 - dones)

    # temporal difference loss = Q(s,a) - (reward + discount * maxQ(s',a))
    loss = criterion(predictions, labels)
    return loss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).
    
    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the 
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0
        finish_x = env.finish_position.x
        finish_y = env.finish_position.y
        max_distance = len(env.lanes) + env.width
        prev_distance = max_distance

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Reward shaping (Manhattan distance)
            agent_x = env.agent.position.x
            agent_y = env.agent.position.y

            # closer to the goal, higher the reward
            distance = abs(agent_x - finish_x) + abs(agent_y - finish_y)
            
            if distance != 0:
              reward = -distance
            # amplify reward when reaching the goal
            else:
              reward = 1000
            
            # made effort in the right direction
            if (distance < prev_distance):
              reward += 30
              prev_distance = distance

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

