#include <stdio.h>
#include <time.h>
#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "coordinates_dataset.hh"

float compute_accuracy(const Matrix& predictions, const Matrix& labels)
