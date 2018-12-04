# Referred to viswanathgs git 'simple_dqn.py' at https://gist.github.com/viswanathgs/abe4a8732a81c666af8bb99254b8e1da
# for defining loss function with target newtwork for DQN 

import gym
import random

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DQN:
    np.random.seed(1673)
    REPLAY_MEMORY_SIZE = 10000    # number of tuples in experience replay
    EPSILON = 1       # epsilon of epsilon-greedy exploation
    EPSILON_DECAY = 0.995   # exponential decay multiplier for epsilon
    MIN_EPSILON = 0.01  # Final minimum value of epsilon in epsilon-greedy
    HIDDEN1_SIZE = 20      # size of hidden layer 1
    HIDDEN2_SIZE = 20      # size of hidden layer 2
    HIDDEN3_SIZE = 20      # size of hidden layer 3
    EPISODES_NUM = 300  # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200      # maximum number of steps in an episode
    LEARNING_RATE = 0.001     # learning rate and other parameters for SGD/RMSProp/Adam
    MOMENTUM = 0.95
    MINIBATCH_SIZE = 32     # size of minibatch sampled from the experience replay
    DISCOUNT_FACTOR = 0.999     # MDP's gamma
    TARGET_UPDATE_FREQ = 200    # number of steps (not episodes) after which to update the target networks
    LOG_DIR = 'tmp/logs'      # directory wherein logging takes place
    LOG_DIR2 = 'tmp/logs2'  # tensorboard --logdir tmp/logs2
    REGULARIZATION_FACTOR = 0.0001
    replay_memory = []

    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        # In case of cartpole, 4 state features
        self.input_size = self.env.observation_space.shape[0]
        # In case of cartpole, 2 actions (right/left)
        self.output_size = self.env.action_space.n
        self.load_model = False
        self.session = tf.Session()

    # Create the Q-network
    def initialize_network(self):

        # placeholder for the state-space input to the q-network
        tf.set_random_seed(1673)
        if self.load_model:
            new_saver = tf.train.import_meta_graph('models/my_model.meta')
            new_saver.restore(self.session,
                              tf.train.latest_checkpoint('models/./'))
            print("Model and data loaded \n")
        else:
            self.x = tf.placeholder(
                tf.float32, [None, self.input_size], name="x")
            self.episode_length = tf.placeholder(
                "float", name="episode_length")

            ############################################################
            # Design your q-network here.
            #
            # Add hidden layers and the output layer. For instance:
            #
            # with tf.name_scope('output'):
            #	W_n = tf.Variable(
            # 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size],
            # 			 stddev=0.01), name='W_n')
            # 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
            # 	self.Q = tf.matmul(h_n-1, W_n) + b_n
            #
            #############################################################

            # Your code here
            with tf.name_scope('hidden1'):
                W1 = tf.Variable(tf.random_uniform(
                    [self.input_size, self.HIDDEN1_SIZE], -1.0, 1.0), name='W1')
                b1 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN1_SIZE], -1.0, 1.0), name='b1')
                h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

            with tf.name_scope('hidden2'):
                W2 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], -1.0, 1.0), name='W2')
                b2 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN1_SIZE], -1.0, 1.0), name='b2')
                h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

            with tf.name_scope('hidden3'):
                W3 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN2_SIZE, self.HIDDEN3_SIZE], -1.0, 1.0), name='W3')
                b3 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN2_SIZE], -1.0, 1.0), name='b3')
                h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

            with tf.name_scope('output'):
                W4 = tf.Variable(tf.random_uniform(
                    [self.HIDDEN3_SIZE, self.output_size], -1.0, 1.0), name='W4')
                b4 = tf.Variable(tf.random_uniform(
                    [self.output_size], -1.0, 1.0), name='b4')
                # , name="Q_values")
                self.Q = tf.squeeze(tf.matmul(h3, W4) + b4)

            self.weights = [W1, b1, W2, b2, W3, b3, W4, b4]

            ############################################################
            # Next, compute the loss.
            #
            # First, compute the q-values. Note that you need to calculate these
            # for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
            #
            # Next, compute the l2 loss between these estimated q-values and
            # the target (which is computed using the frozen target network)
            #
            ############################################################

            # Your code here
            self.targetQ = tf.placeholder(tf.float32, [None])
            self.targetActionMask = tf.placeholder(
                tf.float32, [None, self.output_size])
            q_values = tf.reduce_sum(tf.multiply(
                self.Q, self.targetActionMask), reduction_indices=[1])
            self.loss = tf.reduce_mean(
                tf.square(tf.subtract(q_values, self.targetQ)))

            # # Regularization
            for w in [W1, W2, W3]:
                self.loss += self.REGULARIZATION_FACTOR * \
                    tf.reduce_sum(tf.square(w))

                ############################################################
                # Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam.
                #
                # For instance:
                # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
                # global_step = tf.Variable(0, name='global_step', trainable=False)
                # self.train_op = optimizer.minimize(self.loss, global_step=global_step)
                #
                ############################################################

                # Your code here
                optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        ############################################################

    def train(self, episodes_num=EPISODES_NUM):

        # Initialize the TF session
        saver = tf.train.Saver()
        self.episode_length = 0

        # Initialize summary for TensorBoard
        tf.summary.scalar('loss',self.loss)
        tf.summary.scalar('Episode Length', self.episode_length)
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.session.graph)


        # summary_writer1 = tf.summary.FileWriter(self.LOG_DIR2)
        # summary1 = tf.Summary()

        # Alternatively, you could use animated real-time plots from matplotlib
        # (https://stackoverflow.com/a/24228275/3284912)

        self.session.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        saver.save(self.session, 'models/my_model')

        ############################################################
        # Initialize other variables (like the replay memory)
        ############################################################

        # Your code here
        total_steps = 0
        step_counts = []
        avg_100_steps = []

        # Copying weight to target network
        target_weights = self.session.run(self.weights)

        ############################################################
        # Main training loop
        #
        # In each episode,
        #	pick the action for the given state,
        #	perform a 'step' in the environment to get the reward and next state,
        #	update the replay buffer,
        #	sample a random minibatch from the replay buffer,
        # 	perform Q-learning,
        #	update the target network, if required.
        #
        #
        #
        # You'll need to write code in various places in the following skeleton
        #
        ############################################################

        for episode in range(episodes_num):

            state = self.env.reset()

            ############################################################
            # Episode-specific initializations go here.
            ############################################################
            #
            # Your code here
            #
            self.episode_length = 0
            # done = False

            ############################################################
            # (done == False) and (episode_length < self.MAX_STEPS):
            while True:
                # print(done)
                # for step in range(self.MAX_STEPS):
                ############################################################
                # Pick the next action using epsilon greedy and and execute it
                ############################################################

                # Your code here
                action = None
                if self.EPSILON > random.random():
                    action = self.env.action_space.sample()
                else:
                    q_values = self.session.run(
                        self.Q, feed_dict={self.x: [state]})
                    action = q_values.argmax()

                # Decaying epsilon
                self.EPSILON *= self.EPSILON_DECAY
                if self.EPSILON < self.MIN_EPSILON:
                    self.EPSILON = self.MIN_EPSILON


                ############################################################
                # Step in the environment. Something like:
                # next_state, reward, done, _ = self.env.step(action)
                ############################################################

                # Your code here
                next_state, reward, done, _ = self.env.step(action)

                ############################################################
                # Update the (limited) replay buffer.
                #
                # Note : when the replay buffer is full, you'll need to
                # remove an entry to accommodate a new one.
                ############################################################

                # Your code here
                if done:
                    reward = -100

                self.replay_memory.append(
                    (state, action, reward, next_state, done))

                # removing an entry if replay buffer is full
                if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
                    self.replay_memory.pop(0)

                state = next_state

                ############################################################
                # Sample a random minibatch and perform Q-learning (fetch max Q at s')
                #
                # Remember, the target (r + gamma * max Q) is computed
                # with the help of the target network.
                # Compute this target and pass it to the network for computing
                # and minimizing the loss with the current estimates
                #
                ############################################################

                # Your code here
                if len(self.replay_memory) >= self.MINIBATCH_SIZE:
                    minibatch = random.sample(
                        self.replay_memory, self.MINIBATCH_SIZE)
                    next_states = [s[3] for s in minibatch]

                    feed_dict = {self.x: next_states}
                    feed_dict.update(zip(self.weights, target_weights))
                    q_values = self.session.run(self.Q, feed_dict=feed_dict)
                    max_q_values = q_values.max(axis=1)

                    # Computing Target Q Values
                    target_q = np.zeros(self.MINIBATCH_SIZE)
                    target_action_mask = np.zeros(
                        (self.MINIBATCH_SIZE, self.output_size), dtype=int)
                    for i in range(self.MINIBATCH_SIZE):
                        st, action, reward, n_st, completed = minibatch[i]
                        target_q[i] = reward
                        if not completed:
                            target_q[i] += self.DISCOUNT_FACTOR*max_q_values[i]
                        target_action_mask[i][action] = 1

                    # Performing gradient descent
                    states = [s[0] for s in minibatch]
                    feed_dict = {
                        self.x: states,
                        self.targetQ: target_q,
                        self.targetActionMask: target_action_mask,
                    }
                    _, summary2 = self.session.run(
                        [self.train_op, self.summary], feed_dict=feed_dict)
                    
                    ############################################################
                    # Update target weights.
                    #
                    # Something along the lines of:
                    # if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    # 	target_weights = self.session.run(self.weights)
                    ############################################################

                    # Your code here
                    if total_steps % self.TARGET_UPDATE_FREQ == 0:
                        target_weights = self.session.run(self.weights)
                        print("Targets updated \n")

                    if total_steps % 100 == 0:
                        self.summary_writer.add_summary(summary2, episode)


                        # Save the variables to disk.
                        # save_path = saver.save(self.session, "/models/model.ckpt")
                        # print("Model saved in path: %s" % save_path)


                ############################################################
                # Break out of the loop if the episode ends
                #
                # Something like:
                # if done or (episode_length == self.MAX_STEPS):
                # 	break
                #
                ############################################################

                # Your code here
                total_steps += 1
                self.episode_length += 1

                if done or (self.episode_length == self.MAX_STEPS):
                    break

            ############################################################
            # Logging.
            #
            # Very important. This is what gives an idea of how good the current
            # experiment is, and if one should terminate and re-run with new parameters
            # The earlier you learn how to read and visualize experiment logs quickly,
            # the faster you'll be able to prototype and learn.
            #
            # Use any debugging information you think you need.
            # For instance :

            step_counts.append(self.episode_length)
            mean_steps = np.mean(step_counts[-100:])
            avg_100_steps.append(mean_steps)
            print(
                "Training: Episode = %d, Length = %d, Global step = %d, Last-100 mean steps = %d"
                % (episode, self.episode_length, total_steps, mean_steps))
            # self.summary(value=[tf.Summary.Value(tag="episode length", simple_value=episode_length),])
            # self.summary_writer.add_summary(self.summary, episode)

            # summary1(value=[
            #     tf.Summary.Value(
            #         tag="episode length", simple_value=episode_length),
            # ])
            # summary_writer1.add_summary(summary2, episode)

        saver.save(self.session, 'models/my_model')
        return avg_100_steps, step_counts

    # Simple function to visually 'test' a policy
    def playPolicy(self):
        # Restore variables from disk.
        # saver = tf.train.Saver()
        # saver.restore(self.session, "/models/model.ckpt")
        # print("Model restored.")

        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            # self.env.render()
            q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1

        return steps

    # Simple function to visually 'test' a policy
    def playPolicyFromSavedModel(self):
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(self.session, "/models/model.ckpt")
        print("Model restored.")
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.Q = graph.get_tensor_by_name("Q_values:0")
        # self.Q = tf.add(tf.matmul(h2, W3), b3)
        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            # self.env.render()
            q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1

        return steps


if __name__ == '__main__':

    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    # dqn.load_model = True
    dqn.initialize_network()

    print("\nStarting training...\n")
    avg_100_steps, step_counts = dqn.train()
    print("\nFinished training...\nCheck out some demonstrations\n")

    # Visualize the learned behaviour for a few episodes
    results = []
    for i in range(100):
        episode_length = dqn.playPolicy()
        print("Test steps = ", episode_length)
        results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")

    # Plotting Average total reward of last 100-episodes
    x = np.arange(dqn.EPISODES_NUM)
    plt.clf()
    plt.plot(x, avg_100_steps)
    plt.title("Average total reward of last 100-episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average total reward of last 100-episodes")
    plt.xlim((0, dqn.EPISODES_NUM))
    plt.savefig('Avg_rewards.png', dpi=300)

    # Plotting Episode Lengths
    plt.clf()
    plt.plot(x, step_counts)
    plt.title("Episode Lengths")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Length")
    plt.xlim((0, dqn.EPISODES_NUM))
    plt.savefig('Episode_lengths.png', dpi=300)

    # Plotting Learned agent episodes length for 100 plays
    x = np.arange(100)
    plt.clf()
    plt.plot(x, results)
    plt.title("Learned agent episodes length for 100 plays")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Length")
    plt.xlim((0, 100))
    plt.savefig('Learned_Episode_lengths.png', dpi=300)
