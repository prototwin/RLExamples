<div align="center">
  <img width="3000" height="1000" alt="prototwin" src="https://github.com/user-attachments/assets/0cb6efd9-0685-4ca1-b608-e95aba4f68c5" />
</div>

<br>

<div align="center">
  <a href="https://pypi.org/project/prototwin/"><img src="https://img.shields.io/pypi/v/prototwin.svg" alt="PyPI Version"/></a>
  <a href="https://pypi.org/project/prototwin-gymnasium/"><img src="https://img.shields.io/pypi/v/prototwin-gymnasium.svg" alt="PyPI Version"/></a>
  <a href="https://github.com/prototwin/RLExamples/stargazers"><img src="https://img.shields.io/github/stars/prototwin/RLExamples.svg?style=social&label=Stars" alt="GitHub Stars"/></a>
  <a href="https://twitter.com/prototwin"><img src="https://img.shields.io/twitter/follow/prototwin?style=social" alt="Follow on Twitter"/></a>
  <a href="https://linkedin.com/company/prototwin"><img src="https://img.shields.io/badge/LinkedIn-0077B5?logo=LinkedIn" alt="Follow on LinkedIn"/></a>
  <a href="https://youtube.com/@prototwin"><img src="https://img.shields.io/badge/YouTube-red?logo=youtube&logoColor=white" alt="Subscribe on YouTube"/></a>
</div>

<h4 align="center">

[Website](https://prototwin.com/) | [Documentation](https://prototwin.com/docs) | [Community Forum](https://community.prototwin.com/) | [Sign Up](https://prototwin.com/account/signup)

</h4>

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>ProtoTwin Reinforcement Learning Examples</h1>
    </summary>
  </ul>
</div>

---

ProtoTwin is an online simulation software for robotics and industrial automation. This repository provides a set of examples that you can use with ProtoTwin Connect to train RL agents. To run any of these example models, you will first need to create a ProtoTwin [account](https://prototwin.com/account/signup) and then download the ProtoTwin Connect desktop application. The application supports the following OS:

- Windows 10/11 on x64 and Arm
- MacOS on Apple Silicon
- Linux on x64

## Clone Repository

Clone the repository by running the following command:

```
git clone https://github.com/prototwin/RLExamples.git
```

## Install Dependencies

You need to install the following Python packages to run these examples:

```
pip install prototwin-gymnasium
pip install tensorboard
```

The [prototwin package](https://pypi.org/project/prototwin/) provides a client for starting and connecting to an instance of ProtoTwin Connect. This client may be used to issue commands to load a ProtoTwin model, step the simulation forward in time by one time-step, read signal values and write signal values.
The [prototwin gymnasium package](https://pypi.org/project/prototwin-gymnasium/) provides vectorized and non-vectorized base environments for Gymnasium.

## Examples

### Cartpole V1

ProtoTwin single inverted pendulum (cartpole) tasked with swinging the pole up and balancing it.

![CartPoleV1](https://github.com/user-attachments/assets/ef6117c0-356e-498d-a78d-5cefa474f02e)

To train the cartpole-v1 model you just need to run the following command:
```
cd cartpole-v1
python cartpole-v1.py
```

### Cartpole V2

ProtoTwin cartpole v2 has reduced acceleration in the cart and domain randomization compared to v1 (designed for sim-to-real).

![CartPoleV2](https://github.com/user-attachments/assets/51b73b06-10ff-4ede-9715-a0372c095cc2)

To train the cartpole-v2 model you just need to run the following command:
```
cd cartpole-v2
python cartpole-v2.py
```

### Bipedal

ProtoTwin bipedal robot tasked with walking forward.

![Bipedal](https://github.com/user-attachments/assets/7c9b973f-2bd4-4dbb-8947-a962fb8ebf23)

To train the bipedal model you just need to run the following command:
```
cd bipedal
python bipedal.py
```

### Handstand

Unitree Go2 quadruped robot tasked with performing a handstand.

<img src="https://github.com/user-attachments/assets/6d6cad5f-826c-4385-8694-3a905b325127" alt="Handstand" width="1000" />

To train the handstand model you just need to run the following command:
```
cd handstand
python handstand.py
```

### PingPong

uFactory xArm 6 robot tasked with continuously bouncing a ping pong ball.

<img src="https://github.com/user-attachments/assets/b8773a11-5505-4754-8703-030c6b92b8ef" alt="PingPong" width="1000" />

To train the pingpong model you just need to run the following command:
```
cd pingpong
python pingpong.py
```
