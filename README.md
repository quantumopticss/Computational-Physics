# Computational-Physics
Zifeng Li's code for **Computational Physics(计算物理)**,
you can find some classic topics, including **OdeSolver， Numerical Integral， Interpolation，PDE Equation from Ode** Here,
Some interesting problems like **Ising Model, Kuramoto Model, DLA Model, ...** are also up for grabs.

# What you can get?
You can **Learn Basic Coding Skills** and use **dynamic Programming Language(Python) to solve practial problems**, you will learn how to
using powerful tools including **numpy, scipy** and useful visualization tool including **matplotlib, seaborn**. If you prefer, you can build fundamental tools like **fft, odesolver** for yourself.

# What the code is:
This code base privide **some coding results** from homeworks. Let's use ode (ordinary derivative equation, 常微分方程) as an example : 
usually you will be told Eular methods, backward Eular method. But do you know why these methods can work and what do they do?
$$
y_{n+1} = y_n + f(y_n,x_n) * dx \hspace{0.5cm} \text{Eular method}
\\
y_{n+1} = y_n + f(y_{n+1},x_{n+1}) * dx \hspace{0.5cm} \text{Backward Eular method}
\\
y_{n+1} = y_n + \frac{1}{2} \left[f(y_n,x_n) + f(y_n+f(y_n,x_n)*dx,x_n)\right] * dx \hspace{0.5cm} \text{Two Step Eular method}
\\
$$

The Reason that you will learn is that all these methods are different approaching of Taylor Series (泰勒级数)

# How to use:
You are strongly suggested learning and rebuilding this code base, when you are learning Computational Physica(计算物理)
Just follow the classes, and rebuild what the teacher told you, and compair what you do with this codebase


