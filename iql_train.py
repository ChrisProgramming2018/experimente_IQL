from replay_buffer import ReplayBuffer
from iql_agent import Agent
import sys

import time


def time_format(sec):
    """
    
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train(env, config):
    """

    """
    t0 = time.time()
    memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(state_size=8, action_size=4,  config=config) 
    memory.idx = config["idx"] 
    #for i in range(10):
    #    print("state", memory.obses[i])
    # sys.exit()
    print("memroy idx ",memory.idx)
    if config["mode"] == "predict":
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
            print(text, end = '')
            agent.learn_predicter(memory)
            if t % 2000 == 0:
                # agent.test_predicter(memory)
                agent.save("pytorch_models-{trained_predicter}/")
        return

    
    if config["mode"] == "iql":
        agent.load("pytorch_models-{trained_predicter}-100/")
        agent.test_predicter(memory)
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
            print(text, end = '')
            agent.learn(memory)
            if t % 50 == 0:
                print(text)
                agent.test_predicter(memory)
                #agent.test_q_value(memory)
            if t % 100 == 0:
                for i in range(5):
                    agent.test_policy()

    if config["mode"] == "dqn":
        print("mode dqn")
        agent.dqn_train()
        return
