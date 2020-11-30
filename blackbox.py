import numpy as np
import os
import random
from Semestralka.f10 import f10RaceCarEnv


# This script will run correctly assuming you have pythonpath pointing to the src directory. If you open a pycharm project inside the hw_1 file from sources then pythonpath should be set automatically
# Otherwise you can export PYTHONPATH to add the src directory in the terminal as follows: export PYTHONPATH=${PYTHONPATH}:/home/.../hw_1

class PolicyNet():
    def __init__(self, obs_dim, act_dim):
        self.w = np.zeros((act_dim, obs_dim))
        self.b = np.zeros(act_dim)
        self.obsDim = obs_dim
        self.actDim = act_dim
        # Initialize your weight parameters of your policy.
        # The simplest policy to implement would be a simple affine mapping y = xw+b, where x is the input,
        # w is the weight matrix and b is the bias

    def set_params(self, w):
        # This function takes in a list w and maps it to your policy parameters.
        # The simplest way is to probably make an array out of the w vector and reshape it appropriately
        dim = self.obsDim*self.actDim
        self.b = w[dim:]
        self.w = np.reshape(w[:dim], (self.actDim, self.obsDim))

    def get_params(self):
        # This function returns a list w from your policy parameters. You can use numpy's flatten() function
        #dim = self.obsDim * self.actDim
        #w1 = np.reshape(self.w, dim)
        #w1 = np.append(w1, self.b)
        #w = w1.flatten(order='C')
        w = np.concatenate((self.w.flatten(), self.b.T))
        return w

    def forward(self, x):
        # print("w=",self.w)
        a = self.w @ x + self.b
        #print(a,self.w,self.b,x)
        # Performs the forward pass on your policy. Maps observation input x to action output a
        return a


def f_wrapper(env, policy):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()
        # Map the weight vector to your policy
        policy.set_params(w)
        act=[0,0]
        vel=[0,0]
        while not done:
            # Get action from policy
            act[0],act[1],vel[0],vel[1] = policy.forward(obs)
            vel = [10,10]
            act[1] = act[0]
            # Step environment
            obs, rew, done = env.step(act, vel)
            #print(obs, rew, done)

            reward += rew
        return reward

    return f


def my_opt(f, w_init, iters):
    # Your optimization algorithm. Takes in an evaluation function f, and initial solution guess w_init and returns
    # parameters w_best which 'solve' the problem to some degree.
    w_best = w_init
    w = w_init
    # curr_rew = f(w)
    # r_best = curr_rew
    r_best = 0
    # curr_rew = 0
    for i in range(iters):
        w = np.copy(w_best)
        for j in range(6):
            w[j] = w_best[j] + random.gauss(0, 0.05)
        curr_rew = f(w)
        if curr_rew > r_best:
            w_best = w
            r_best = curr_rew
            # print(r_best)

    return w_best, r_best


def test(w_best, max_steps=70, animate=False):
    # Make the environment and your policy
    env = f10RaceCarEnv(animate=animate, max_steps=max_steps)
    policy = PolicyNet(env.obs_dim, env.act_dim)

    # Make evaluation function
    f = f_wrapper(env, policy)

    # Evaluate
    r_avg = 0
    eval_iters = 10
    for i in range(eval_iters):
        r = f(w_best)
        r_avg += r
        print(r)
    return r_avg / eval_iters


if __name__ == "__main__":
    train = True
    train = False
    policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'f10_bbx.npy')
    max_steps = 1000
    N_training_iters = 100
    w_best = None

    if train:
        # Make the environment and your policy
        env = f10RaceCarEnv(animate=False, max_steps=max_steps)
        policy = PolicyNet(env.obs_dim, env.act_dim)

        # Make evaluation function
        f = f_wrapper(env, policy)

        # Initial guess of solution
        w_init = policy.get_params()

        # Perform optimization
        w_best, r_best = my_opt(f, w_init, N_training_iters)

        print(f"r_best: {r_best}")

        # Save policy
        np.save(policy_path, w_best)
        env.close()

    if not train:
        w_best = np.load(policy_path)
    print(f"Avg test rew: {test(w_best, max_steps=max_steps, animate=not train)}")



