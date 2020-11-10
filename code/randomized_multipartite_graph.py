import numpy as np
import csv
import random


def randomize_multipartite_graph(number_of_files, number_of_edges, number_of_vertices):

    for file_number in range(number_of_files):
        with open(f'Graph_{file_number}_edges_{number_of_edges}', 'w', newline='') as file:
            for i in range(number_of_edges):
                writer = csv.writer(file)
                writer.writerow([random.randint(1, number_of_vertices), random.randint(1, number_of_vertices),
                                 np.round(random.uniform(0, 100), 2)])

            file.close()


def main():
    randomize_multipartite_graph(3, 10000000, 9000)
    randomize_multipartite_graph(3, 100000000, 90000)


if __name__ == '__main__':
    main()
