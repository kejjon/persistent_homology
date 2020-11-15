from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from random import gauss
from ripser import ripser
from persim import plot_diagrams
from sklearn import datasets
import setup

plt.style.use('ggplot')

SPACE = setup.space
NUMBER_OF_POINTS = setup.number_of_points

def generate_xy(lower_x, upper_x, lower_y, upper_y, number_of_points):
    x_list = np.random.uniform(lower_x, upper_x, number_of_points)
    y_list = np.random.uniform(lower_y, upper_y, number_of_points)
    return x_list, y_list

def generate_half_s2(x, y):
    z_list = np.sqrt(np.abs(1 - x*x - y*y))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z_list, c='g', marker='o')
    plt.show()
    return np.sqrt(np.abs(1 - x*x - y*y))


def project_on_planes(plane, x, y, z):
    zeroPlane = [0 for i in range(NUMBER_OF_POINTS)]
    if plane=="xy":
        z = zeroPlane
    elif plane=="xz":
        y = zeroPlane
    else:
        x = zeroPlane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def generate_list(func, lst):
    return [func(x) for x in lst]


def generate_rand_list(func, array):
    func_array = func(array)
    for i in range(len(func_array)):
        func_array[i] = func_array[i] * random.uniform(0.5, 1.5)
    return func_array

def generate_pt_set(*args):
    point_list = []
    for i in range(len(args)):
        point_list.append(args[i])
    point_tuple = tuple(point_list)
    return np.vstack(point_tuple).T

def generate_pt_set1(*args):
    point_list = []
    for i in range(len(args[0])):
        point = []
        for j in range(len(args)):
             point.append(args[j][i])
        point_list.append(point)
    return point_list

def enterUserMatrix():

    R = int(input("Enter the number of rows: "))
    C = int(input("Enter the number of columns: "))

    print("Enter the entries in a single line (separated by space): ")

    # User input of entries in a
    # single line separated by space
    entries = list(map(int, input().split()))

    # For printing the matrix
    matrix = np.array(entries).reshape(R, C)
    print(matrix)


def main():
    # enterUserMatrix()
    global points_in_space
    x, y = generate_xy(setup.lower_x, setup.upper_x, setup.lower_y, setup.upper_y, NUMBER_OF_POINTS)
    cosx_values = generate_rand_list(np.cos, x)
    sinx_values = generate_rand_list(np.sin, x)
    cosy_values = generate_rand_list(np.cos, y)
    siny_values = generate_rand_list(np.sin, y)
    y_random = generate_rand_list(lambda x : x, y)
    sinx_cosy = sinx_values*cosy_values
    sinx_siny = sinx_values*siny_values

    X, Y, Z = 0, 0, 0


    if SPACE == "SPHERE_3":
        X = sinx_cosy
        Y = sinx_siny
        Z = cosx_values
        points_in_space = generate_pt_set(X, Y, Z)
    elif SPACE == "CYLINDER_3":
        X = cosx_values
        Y = sinx_values
        Z = y_random
        points_in_space = generate_pt_set(X, Y, Z)
    elif SPACE == "TORUS_3":
        X = (setup.upper_r + setup.lower_r*cosx_values)*cosy_values
        Y = (setup.upper_r + setup.lower_r*cosx_values)*siny_values
        Z = setup.lower_r*sinx_values
        points_in_space = generate_pt_set(X, Y, Z)
    elif SPACE == "TORUS_4":
        points_in_space = generate_pt_set(cosx_values, sinx_values, cosy_values, siny_values)
    elif SPACE == "KLEIN_BOTTLE_4":
        X = (setup.upper_r + setup.lower_r * cosx_values) * cosy_values
        Y = (setup.upper_r + setup.lower_r * cosx_values) * siny_values
        Z = setup.lower_r * sinx_values * generate_rand_list(np.cos, y/2)
        T = setup.lower_r * sinx_values * generate_rand_list(np.sin, y/2)
        points_in_space = generate_pt_set(X, Y, Z, T)
    elif SPACE == "PRO_PLANE_4" or SPACE == "PRO_PLANE_6":
        square_x = sinx_cosy * sinx_cosy
        square_y = sinx_siny * sinx_siny
        xy = sinx_cosy*sinx_siny
        xz = sinx_cosy*cosx_values
        yz = sinx_siny*cosx_values
        if SPACE == "PRO_PLANE_4":
            points_in_space = generate_pt_set(square_x - square_y, xy, xz, yz)
        else:
            square_z = cosx_values * cosx_values
            points_in_space = 1 / (square_x + square_y + square_z) * \
                              generate_pt_set(square_x, square_y, square_z, xy, xz, yz)


    print(points_in_space)

    if setup.is_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X, Y, Z, c='g', marker='o')
        plt.show()

    # project_on_planes("xy", X, Y, Z)
    # project_on_planes("yz", X, Y, Z)
    # project_on_planes("xz", X, Y, Z)

    # data = np.array([np.array(xi) for xi in points_in_space])

    dgms = ripser(points_in_space, maxdim=2, thresh=1.2)['dgms']
    plot_diagrams(dgms, title=SPACE, colormap='seaborn', show=True)
    # plot_diagrams(dgms, title=SPACE + ' H0', colormap='ggplot', plot_only=[0], ax=plt.subplot(131))
    # plot_diagrams(dgms, title=SPACE + ' H1', colormap='ggplot', plot_only=[1], ax=plt.subplot(132))
    # plot_diagrams(dgms, title=SPACE + ' H2', colormap='ggplot', plot_only=[2], ax=plt.subplot(133))
    # plt.show()
    # plot_diagrams(dgms, show=True)
    # plot_diagrams(dgms, colormap='seaborn-darkgrid', lifetime=True, show=True)

main()
