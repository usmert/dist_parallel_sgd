#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;

void train(double* dataset, double* w, int rank, int num_procs) {
    double prev_loss = 10.0;
    double diff = 1.0;
    int epoch = 0;
    int samples_per_proc = (585024 + num_procs - 1) / num_procs;
    int start_idx = min(rank * samples_per_proc, 585024);
    int end_idx = min(start_idx + samples_per_proc, 585024);

    while (prev_loss > 6.75) {

        double loss = 0.0;

        for (int i = start_idx; i < end_idx; i += 8192) {
            double* grad = new double[27];
            for (int ii = i; ii < min(i+8192, end_idx); ++ii) {
                double y = dataset[ii*28 + 27];
                double y_hat = 0.0;
                for (int j = 0; j < 27; ++j) {
                    y_hat += dataset[ii*28 + j] * w[j];
                }
                double diff = y_hat - y;
                loss += diff * diff;
                for (int j = 0; j < 27; ++j) {
                    grad[j] += 2 * diff * dataset[ii*28 + j];
                }
            }

            for (int j = 0; j < 27; ++j) {
                w[j] -= 0.00000001 * (grad[j] / 8192);
            }
            delete[] grad;
        }
        if (start_idx < end_idx) {
            loss = loss / 585024;
        }
        double recv_loss = 0.0;
        MPI_Allreduce(&loss, &recv_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double* recv_w = new double[27];
        MPI_Allreduce(w, recv_w, 27, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int j = 0; j < 27; ++j) {
            w[j] = recv_w[j] / num_procs;
        }
        diff = abs(prev_loss - recv_loss);
        prev_loss = recv_loss;
        if (rank == 0) {
            cout << "Epoch: " << epoch << " Loss: " << loss << endl;
        }
        epoch++;
    }
}