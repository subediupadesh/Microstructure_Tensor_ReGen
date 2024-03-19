import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(theta, c):
    return 1 / (1 + np.exp(-theta * (2 * c - 1)))

def plot_function(thetas, colors, c_values):
    fig, ax = plt.subplots()
    for i, (theta, color) in enumerate(zip(thetas, colors)):
        h_values = sigmoid(theta, c_values)
        ax.plot(c_values, h_values, label=f'$\\theta_{i+1}$ = {theta}', color=color)

    ax.set_xlabel('c')
    ax.set_ylabel('h')
    ax.set_title('Sigmoid Function for Different Thetas')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title('Sigmoid Function Plotter')
    thetas = []
    colors = []
    for i in range(5):
        theta = st.slider(f'$\\theta_{i+1}$', min_value=5, max_value=50, value=10)
        thetas.append(theta)
        color = st.color_picker(f'Choose Color for Theta_{i+1}')
        colors.append(color)

    c_values = np.linspace(0, 1, 100)

    plot_function(thetas, colors, c_values)

if __name__ == '__main__':
    main()

