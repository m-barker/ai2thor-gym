# ai2thor-gym
Gymnasiumm wrapper for ai2thor, you can test this by running `python3 ai2thor-gym-wrapper.py`, which will showcase an agent taking random actions in a kitchen environment, with rewards given for progress
made towards the `CookEgg` task, in which the agent must find, crack, and cook an egg using the stove
or the microwave. This uses the standard (older) gym API:

```python
env = CookEggEnv()
done = False
while not done:
    action = np.zeros(env.action_space.n)
    int_action = env.action_space.sample()
    action[int_action] = 1
    obs, reward, done, info = env.step(action)
    if reward > -1.0:
        print(f"Reward: {reward}")
```

Actions that interact with objects have been modified to simply interact with the neareast reachable valid object to the agent, to prevent the need of parameterising the actions.
